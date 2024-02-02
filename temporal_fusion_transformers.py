#!/usr/bin/env python
# coding: utf-8

# This notebook describes the temporal fusion transformers [@lim2021temporal] architecture, and ports it over to keras 3 while making some punctual improvements, including bringing the notation closer to the one in the paper.
# 
# The original repository is [here](https://github.com/google-research/google-research/tree/master/tft).

# In[1]:


#| output: false

from __future__ import annotations

import os
import torch
os.environ["KERAS_BACKEND"] = "torch"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras
from keras import layers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from datetime import timedelta
from dateutil.relativedelta import relativedelta
from fastcore import docments
from nbdev.showdoc import show_doc
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# # Introduction

# The main characteristics of TFT that make it interesting for nowcasting or forecasting purposes are:
# 
# - **multi-horizon forecasting**: the ability to output, at each point in time $t$, a sequence of forecasts for $t+h, h > 1$
# - **quantile prediction**: each forecast is accompanied by a quantile band that communicates the amount of uncertainty around a prediction
# - **flexible use of different types of inputs**: static inputs (akin to fixed effects), historical input and known future input (eg, important holidays, years that are known to have major sports events such as Olympic games, etc)
# - **interpretability**: the model learns to select variables from the space of all input variables to retain only those that are globally meaningful, to assign attention to different parts of the time series, and to identify events of significance
# 
# ## Main innovations
# 
# The present model includes the following innovations:
# 
# - **Multi-frequency input**
# 
# - **Context enhancement from lagged target**: the last known values of the target variable are embedded (bag of observations), and this embedding is used similar to the static context enhancement as a starting point for the cell value in the *decoder* LSTM.

# # Preparing the data
# 
# The functions below will be tested with simulated and real data. The former helps to illustrate issues like dimensions and overall behaviour of a layer, whereas the latter will demonstrate on a real setting in economics how the input and output data relate to one another.
# 
# More specifically, the real data used will be a daily nowcasting exercise of monthly inflation. Note that the models will not necessarily perform well, since their use here is for illustration purposes and thus they are not optimised. Also, the dataset is not an ideal one for nowcasting: other variables could also be considered.

# ## Download economic data
# 
# This is a panel dataset. In addition to the time dimension, it can contain any number of categorical dimensions - for example, combine country and sector.

# In[2]:


#| warning: false

df_all = pd.read_csv("data/nowcast_dataset_complete_Jan-24-2024.csv")
df_all['index'] = pd.to_datetime(df_all['index'])


# ## Prepare data

# In[3]:


countries = ['CA', 'CH', 'DE', 'FR', 'GB', 'IN', 'JP', 'US']
columns = ["equity", "fx", "Cab", "ShortGovYield", "LongGovYield", "UnempRate", "neer", "policyRate", "energy", "food", "metal"]

filter_freq_d = df_all['frequency'] == 'd'
filter_cty = df_all['country'].isin(countries)
filter_dates = df_all['index'] >= '1980-01-01'


# In[4]:


df_input = df_all[filter_freq_d & filter_cty & filter_dates].copy()
df_input.drop (['Unnamed: 0', 'frequency'], axis=1, inplace=True)
df_input.set_index(['index', 'country'], inplace=True)
df_input = df_input.loc[:, columns]
df_input = df_input.unstack('country')
df_input.columns = ['__'.join(col).strip() for col in df_input.columns.values]
df_input.dropna(how='all', inplace=True)


# In[5]:


target_var = 'CPIh'
df_target = df_all.loc[
    (df_all['frequency'] == 'm') & (df_all['country'].isin(countries)) & filter_dates,
    ['index', 'country'] + [target_var]
] \
    .set_index(['index', 'country']) \
    .unstack('country') \
    .droplevel(0, axis=1) \
    .dropna()


# In[6]:


df_target_12m_pct = (100 * df_target.pct_change(12))
df_target_1m_pct = (100 * df_target.pct_change(1))


# In[7]:


#| fig-align: center

ax = df_target_12m_pct.plot(figsize=(20, 6))
ax.axhline(y=0, color='black')
ax.set_title("Inflation", fontsize=16, fontstyle='italic')
plt.show()


# ## Date features
# 
# In addition to data provided by the user, the model automatically loads categorical features related to each date.

# In[8]:


def date_features(
    date_range, # Range of dates for which to create date features
    is_monthly:bool=False # Is the date measured at the monthly frequency?
)->pd.DataFrame: # Categorical date features
    "Categorical features for each day in a range of dates"
    if is_monthly:
        return pd.DataFrame({
        'Month of Year': date_range.month
    })
    else:
        return pd.DataFrame({
            'Day of Week': date_range.dayofweek + 1, # This is the only date feature with zeros, which are masked out
            'Day of Month': date_range.day,
            'Day of Year': date_range.dayofyear,
            'Week of Month': (date_range.day - 1) // 7 + 1,
            'Week of Year': pd.Index(date_range.isocalendar().week).astype('int32'),
            'Month of Year': date_range.month
        })


# ### Combining the date features in the main data

# In[11]:


date_feat = date_features(df_input.index)
date_feat.index = df_input.index
df_input = pd.concat([df_input, date_feat], axis=1)


# ## Splitting the dataset
# 
# This step needs to be done somewhat differently than normal datasets: In the validation and test periods, the input data can very well be from the previous chunk (training and validation, respectively).
# 
# In practice, this means that we just need to split the **dates** at which the nowcasting will take place. Then, a data loading function (see @sec-dataloaders) can read from the whole time series to look back from that date as needed according to the desired time window.

# In[12]:


start_date = df_input.index.min()
end_date = df_input.index.max()
cutoff_date_train = pd.to_datetime('2005-01-01')
cutoff_date_valid = pd.to_datetime('2020-01-01')

dates_train = pd.date_range(start_date, cutoff_date_train - timedelta(days=1))
dates_valid = pd.date_range(cutoff_date_train, cutoff_date_valid - timedelta(days=1))
dates_test = pd.date_range(cutoff_date_valid, end_date)


# Now all input variables that are not available in the training dataset are removed, and only those with at least some information are kept.

# In[13]:


keep_cols = df_input.loc[min(dates_train):max(dates_train)].dropna(axis=1, how='all').columns
df_input = df_input[keep_cols]


# ## Identifying continuous and categorical variables {#sec-contcat}
# 
#  The model distinguishes continuous from categorical variables if the user does not provide a list of variable names in a simple (simplistic) way: integer-valued variables that start with one are considered categorical, all other are continuous.
# 
# The criteria that categorical variables start with one is to ensure that the user does not unwarrantedly pass on categorical variables with zero, since zeros are considered to be a padding for variable-length input data.
# 
# For variables that are naturally valued in integers, such as the count of number of firms, etc, the user can either ensure there is a zero amongst the integer at any time of the **training** input data, or convert these values to floating numbers. Another alternative that might be relevant in some cases is to use the natural logarithm of that data.

# In[14]:


int_cols = df_input.select_dtypes(include=['int']).columns
float_cols = df_input.select_dtypes(include=['float']).columns

# Columns that are float but actually contain integer values starting with one
cat_cols = []

for col in int_cols:
    if min(df_input[col]) > 0:
        cat_cols.append(col)

for col in float_cols:
    if (df_input[col] % 1 == 0).all() and min(df_input[col]) > 0:  # Check if the fractional part is 0 for all values and the lowest integer is 1
        cat_cols.append(col)

cont_cols = [c for c in df_input.columns if c not in cat_cols]

assert len(cont_cols) + len(cat_cols) == df_input.shape[1]


# Further, the categorical variables require a dictionary that indicates the cardinality of each variable.

# ## Scaling the continuous variables
# 
# The input series need to be scaled, according to the training data mean and standard deviation.
# 
# The target series will not be scaled because it is already a small number close to zero that is not exploding in nature.

# In[15]:


#| warning: false

scl = StandardScaler()
scl.fit(df_input.loc[dates_train.min():dates_train.max(), cont_cols])
df_input_scl = pd.DataFrame(
    scl.transform(df_input[cont_cols]),
    index=df_input.index,
    columns=cont_cols
    )
df_input_scl = pd.concat([df_input_scl, date_feat], axis=1)

assert df_input_scl.shape == df_input.shape


# ## Cardinality of categorical variables
# 
# Each categorical variable has its own cardinality. This value is important when creating the embedding layer for each variable; see @sec-input.

# In[16]:


cardin_hist = {c: len(df_input_scl[c].unique()) + 1 for c in cat_cols}

cardin_hist


# The cardinality of the static variable(s) must also be included:

# In[17]:


cardin_stat = dict(Countries=len(df_target.columns) + 1)


# The country list also requires an encoding/decoding dictionary for subsequent analyses.

# In[18]:


country_enc_dict = {cty: idx + 1 for idx, cty in enumerate(df_target.columns)}
country_dec_dict = {idx: cty for cty, idx in country_enc_dict.items()}


# ## Dealing with missing data
# 
# 
# Missing data is dealt with by replacing `NaN`s in the input data with zeros. This has two effects:
# 
# * it prevents embedding categorical variables since zeros are masked out
# 
# * for the continuous data, the input layer weights do not pick up any information, and the constant (or "bias" in machine learning language) is responsible for conveying any information to subsequent neurons.
# 
# A more sophisticated approach would be to estimate missing data based on other contemporaneous and past data. For simplicity, this approach will not be followed in this example.

# Obviously this step needs to be done after the input data is scaled, otherwise the zeros would be wrongly contributing to the mean and standard deviation values.

# In[19]:


df_input_scl.fillna(0, inplace=True)


# Finally, we remove the input months for which there is no inflation data (eg, due to the one-month growth calculation):

# In[20]:


df_input_scl = df_input_scl[df_target_1m_pct.dropna().index.min():]


# ## Splitting the data

# Now we separate the training, validation and test data.

# In[21]:


df_input_train = df_input_scl[:dates_train.max()]
df_input_valid = df_input_scl[dates_train.max():dates_valid.max()]
df_input_test = df_input_scl[dates_valid.max():dates_test.max()]

df_input_train.shape[0], df_input_valid.shape[0], df_input_test.shape[0]


# ## Data loaders {#sec-dataloaders}

# Ideally a data loader should:
# 
# * create a pair of input/target data
# 
# * the input data should contain:
# 
#     * continuous data from all countries
# 
#     * categorical date features
# 
#     * categorical feature of the country (ie, which country it is)
# 
#     * known future inputs
# 
# On the known future inputs: those will be essentially the categorical date features, broadcasted until the end of the desired month to be nowcasted (ie, either the current month or a future one). The nowcast will then be the value of inflation at the end of the "as-of" month.
# 
# > Note: so far, the only known future data used by the model are the categorical features from the dates up to the last day of the nowcasted/forecasted month. However, an important known future data for inflation are the central bank policy variables. These are not yet dealth with in this code, but a future version should incorporate an intelligent way for users to input a vector of policy variables, with the dates up until which they would be known. This could be done in a separate DataFrame with policy variables, which would arguably facilitate working with this data separately from all others.

# In[22]:


def sample_nowcasting_data(
    df_daily_input:pd.DataFrame, # DataFrame with the time series of the input variables
    df_target:pd.DataFrame, # DataFrame with the time series of the target variable
    min_context:int=90, # minimum context length in number of daily observations
    context_length:int=365, # context length in number of daily observations (leads to padding if not reached in sampling)
    num_months:int=12, # number of months (1 is current month)
    sampled_day:None|str=None, # None (default) randomly chooses a date; otherwise, YYYY-MM-DD date selected a date
    country:None|str=None # None (default) randomly chooses a country; otherwise, 2-digit ISO code selectes a country
):

    # first step: determine the valid dates for sampling
    # from the left, they should allow at least `context_length` up until the sampled date
    # from the right, they should allow enough room to retrieve all of the target inflation months
    # only the dates "in the middle" can be sampled
    all_dates = df_daily_input.index
    earliest_poss_date = all_dates[min_context]

    delta_latest_month = num_months - 1
    latest_poss_date = all_dates.max() - relativedelta(months=delta_latest_month) + pd.offsets.MonthEnd(0)

    dates_for_sampling = df_daily_input.loc[earliest_poss_date:latest_poss_date].index

    # sample a random date, context length and country
    if sampled_day is None:
        sampled_day = pd.to_datetime(np.random.choice(dates_for_sampling))
        sampled_ctl = np.random.randint(low=min_context, high=context_length) \
            if min_context < context_length \
            else context_length
    else:
        # sampled_ctl is the longest possible since setting a date means the data 
        # will not be used for training models but for prediction/evaluation
        sampled_day = pd.to_datetime(sampled_day)
        sampled_ctl = context_length
    X_cat_stat = np.random.randint(low=1, high=len(country_enc_dict)+1, size=(1,)) \
        if country is None \
        else keras.ops.reshape(np.array(country_enc_dict[country]), (1,))

    # create the historical observed data
    earliest_date = sampled_day - relativedelta(days=sampled_ctl - 1)
    df_hist = df_daily_input.loc[earliest_date:sampled_day]
    if df_hist.shape[0] < context_length:
        df_pad = pd.DataFrame(np.zeros((context_length-df_hist.shape[0], df_hist.shape[1])))
        df_pad.columns = df_hist.columns
        df_hist = pd.concat([df_pad, df_hist], axis=0, ignore_index=True)
    X_cont_hist = df_hist[cont_cols].values
    X_cat_hist = df_hist[cat_cols].values

    # create the future known data: month of the year
    # note: any other known future information of interest should be included here as well
    # eg: mon pol committee meeting dates, months of major sports events, etc
    # anything that could influence inflation dynamics
    target_month = (sampled_day + relativedelta(months=num_months)).replace(day=1) \
        if num_months == 1 \
        else [(sampled_day + relativedelta(months=d)).replace(day=1) for d in range(num_months)]
    X_fut = date_features(pd.DataFrame(index=target_month).index, is_monthly=True).values

    # create the target variables
    y = df_target.loc[target_month, country_dec_dict[int(X_cat_stat[0])]].values
    
    X_cat_stat = keras.ops.expand_dims(X_cat_stat, axis=1)
    return [X_cont_hist, X_cat_hist, X_fut, X_cat_stat], y


# Note that `min_context` not necessarily number of days because of weekends, etc.
# 
# In practice, this means that the sampled data almost always needs to be padded, even when `min_context` is equal to `context_length`. See the example below.

# In[27]:


def prepare_data_samples(
    n_samples:int=1000,
    **kwargs
):
    "Transforms the dataset from tabular format to a dataset used for training the TFT model."
    X_cont_hist = []
    X_cat_hist = []
    X_fut = []
    X_cat_stat = []
    y = []
    for i in tqdm(range(n_samples)):
        [indiv_cont_hist, indiv_cat_hist, indiv_fut, indiv_cat_stat], indiv_y = sample_nowcasting_data(**kwargs)
        X_cont_hist.append(indiv_cont_hist)
        X_cat_hist.append(indiv_cat_hist)
        X_fut.append(indiv_fut)
        X_cat_stat.append(indiv_cat_stat)
        y.append(indiv_y)
    
    X_cont_hist = keras.ops.stack(X_cont_hist, axis=0)
    X_cat_hist = keras.ops.stack(X_cat_hist, axis=0)
    X_fut = keras.ops.stack(X_fut, axis=0)
    X_cat_stat = keras.ops.stack(X_cat_stat, axis=0)
    y = keras.ops.stack(y, axis=0)

    return [X_cont_hist, X_cat_hist, X_fut, X_cat_stat], y


# This function serves the purpose to structure the data in a way that the TFT can ingest, while taking advantage of the number of combinations of sampled data X context size to create a larger dataset.

# Note that the argument `max_context_length_days` cannot promise the user to reach the actual number, since there might be weekends or other dates without any information.

# For models that will nowcast/forecast more than one month, the argument `delta_month` needs to be a list or an iterator as `range()`:

# Now the months for which to forecast are February (the current month of the sampled day), March and April, corresponding to the three months indicated in the argument.

# All historical time series in the same batch have the same length. This length varies beetween batches.

# # Architecture
# 
# First, common notation is introduced, and then individual components are presented. At the end of this section, the whole model is put together.

# ## Notation
# 
# * unique entities: $i \in (1, \dots\, I)$
# * time periods $t \in [0, T_i]$
#   * $k \geq 1$ lags
#   * $h \geq 1$ forecasting period
# * set of entity-level static covariates: $s_i \in \mathbf{R}^{m_s}$
# * set of temporal inputs: $\chi_{i, t} \in \mathbf{R}^{m_\chi}$
#   * $\chi_{i,t} = [z_{i,t}, x_{i,t}]$
#     * $z_{i,t} \in \mathbf{R}^{m_z}$ are historical inputs
#     * $x_{i,t} \in \mathbf{R}^{m_z}$ are a priori known inputs (eg, days of the week or years that have major sports events)
#   * $m_\chi$ is the number of total input variables, where $m_\chi = m_z + m_x$
# * target scalars: $y_{i,t}$
#   * $\hat{y}_{i,t,q} = f_q(y_{i,t-k:t}, z_{i,t-k:t}, x_{i,t-k:t+h}, s_i)$
# * hidden unit size (common across all the TFT architecture for convenience): $d_{\text{model}}$
# * transformed input of $j$-th variable at time $t$: $\xi_t^{(j)} \in \mathbf{R}^{d_{\text{model}}}$
#   * $\Xi_t = [\xi_t^{(1)}, \dots, \xi_t^{(m_\chi)}]$

# ## Example data
# 
# All real data examples below use the same data from @sec-real_data.

# ## Dense layer

# One of the fundamental units in the TFT network is the dense layer:
# 
# $$
# \mathbb{Y} = \phi(\mathbf{W} x + \mathbf{b}),
# $$ {#eq-dense}
# 
# where $x$ is the input data and $\mathbb{Y}$ is its output, $\phi$ is an activation function (when it is applied), $\mathbf{W} \in \mathbf{R}^{(d_{\text{size}} \times d_{\text{inputs}})}$ is a matrix of weights and $\mathbf{b} \in \mathbf{R}^{d_{size}}$ is a vector of biases.

# ## Input data transformations {#sec-input}

# > Transforms all input variables into a latent space
# 
# All input data, regardless if historical, static or future, are transformed into a feature representation with dimension $d_{\text{model}}$. In other words, $\chi_{t}^{(j)}$, variable $j$'s observation at each time period $t$, undergoes an injective mapping $f^{(j)} : \mathbb{R} \to \mathbb{R}^{d_{\text{model}}}$ for continuous data and $f^{(j)} : \mathbb{N} \to \mathbb{R}^{d_{\text{model}}}$ for categorical data.
# 
# If the variable $\chi_{t}^{(j)}$ is continuous, this transformation is done by $d_{\text{model}}$ linear regressions, the coefficients of which are determined as part of the neural network training:
# 
# $$
# \xi_t^{(j)} = \mathbf{W}^{(j)} \chi_{t}^{(j)} + \mathbf{b}^{(j)},
# $$
# 
# where $\xi_t^{(j)}, \mathbf{W}^{(j)}, \mathbf{b}^{(j)} \in \mathbb{R}^{d_{\text{model}}}$. Note that $\mathbf{W}^{(j)}, \mathbf{b}^{(j)}$ are the same for variable $j$ at all time periods (ie, the layer is time-distributed).
# 
# Conversely, if the $j^{\text{th}}$ variable is categorical, then the transformation is an embedding. Each embedding layer requires a specific number of different categories, ie the cardinality of the categorical variable. This cardinality is assumed to be stable or decreasing outside of the training period; otherwise a new category would appear at testing time for which the model has not learned an embedding.
# 
# In any case, it is this embedded data, $\xi_t^{(j)}$, that is used in all the next steps of the TFT network.
# 
# See @sec-contcat for details of how the model determines which variables are continuous or categorical.

# In[32]:


#| code-fold: hide

class MultiInputContEmbedding(keras.Layer):
    def __init__(
        self, 
        d_model:int, # Embedding size, $d_\text{model}$
        **kwargs
    ):
        "Embeds multiple continuous variables, each with own embedding space"
        
        super(MultiInputContEmbedding, self).__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        super(MultiInputContEmbedding, self).build(input_shape)

        # input_shape: (batch_size, num_time_steps, num_variables)
        num_variables = input_shape[-1]
        
        self.kernel = self.add_weight(
            shape=(num_variables, self.d_model),
            initializer='uniform',
            name='kernel'
        )

        self.bias = self.add_weight(
            shape=(self.d_model,),
            initializer='zeros',
            name='bias'
        )
    
    def call(
        self,
        inputs # Data of shape: (batch size, num time steps, num variables)
    ):     
        "Output shape: (batch size, num time steps, num variables, d_model)"

        # Applying the linear transformation to each time step
        output = keras.ops.einsum('bti,ij->btij', inputs, self.kernel)
        output += self.bias
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], self.d_model


# In[36]:


#| code-fold: hide

class MultiInputCategEmbedding(keras.Layer):
    def __init__(
        self,
        d_model:int, # Embedding size, $d_\text{model}$
        cardinalities:dict, # Variable: cardinality in training data
        **kwargs
    ):
        "Embeds multiple categorical variables, each with own embedding function"
        super(MultiInputCategEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.cardinalities = cardinalities
        
    def build(self, input_shape):
        super(MultiInputCategEmbedding, self).build(input_shape)
        if len(self.cardinalities.keys()) != input_shape[-1]:
            raise ValueError("`cardinalities` should have as many elements as the input data's variables.")
        
        self.embed_layer = {var:
            keras.Sequential([
                layers.Embedding(
                    input_dim=cardin,
                    output_dim=self.d_model,
                    mask_zero=True,
                    name="input_embed_" + var.replace(" ", "_")
                )
            ]) for var, cardin in self.cardinalities.items()
        }
        super(MultiInputCategEmbedding, self).build(input_shape)

    def call(
        self,
        inputs # Data of shape: (batch size, num time steps, num variables)
    ):
        "Output shape: (batch size, num time steps, num variables, d_model)"
        embeds = [
            self.embed_layer[var](inputs[:,:,idx])
            for idx, var in enumerate(self.cardinalities.keys())
        ]
        return keras.ops.stack(embeds, axis=2) # keras.ops.concatenate(embeds, axis=-1)


# Note that this function masks out the data whenever categories are set to zero. This is to ensure that the model can take in variable-sized inputs. Because of this, the model requires inputted categorical data to be added 1 whenever zero is a possible category.

# > Note to self: I don't like the approach above where we rely implicitly on the input data's ordering to extract the cardinality. But this is a practical way to get things going. It should be changed to a more robust in the future.

# In[43]:


#| code-fold: hide

class InputTFT(keras.Layer):
    def __init__(
        self,
        d_model:int=16, # Embedding size, $d_\text{model}$
        **kwargs
    ):
        "Input layer for the Temporal Fusion Transformer model"
        super(InputTFT, self).__init__(**kwargs)
        self.d_model = d_model
        
        self.flat = layers.Flatten()
        self.concat = layers.Concatenate(axis=2)
    
    def build(self, input_shape):
        self.cont_hist_embed = MultiInputContEmbedding(
            self.d_model,
            name="embed_continuous_historical_vars"
        )
        self.cat_hist_embed = MultiInputCategEmbedding(
            self.d_model, 
            cardinalities=cardin_hist,
            name="embed_categ_historical_vars"
        )
        self.cat_fut_embed = MultiInputCategEmbedding(
            self.d_model, 
            # Note below the same categorical variables are just the months in an year. 
            # This situation may not apply to all cases.
            # More complex models using other categorical future known data might require
            # another cardinalities dictionary.
            cardinalities={'Month of Year': 13},
            name="embed_categ_knownfuture_vars"
        )
        self.cat_stat_embed = MultiInputCategEmbedding(
            self.d_model, 
            cardinalities=cardin_stat,
            name="embed_categ_static_vars"
        )
        super(InputTFT, self).build(input_shape)

    def call(
        self, 
        # List of data with shape: [(batch size, num hist time steps, num continuous hist variables), (batch size, num hist time steps, num categorical hist variables), (batch size, num static variables), (batch size, num future time steps, num categorical future variables)]
        input:list 
    ):
        """List of output with shape: [
            (batch size, num hist time steps, num historical variables, d_model),
            (batch size, num future time steps, num future variables, d_model)
            (batch size, one, num static variables, d_model),
        ]"""
        cont_hist, cat_hist, cat_fut, cat_stat = input
        if len(cat_stat.shape) == 2:
            cat_stat = keras.ops.expand_dims(cat_stat, axis=-1)

        cont_hist = self.cont_hist_embed(cont_hist)
        #cont_hist = keras.ops.swapaxes(cont_hist, axis1=2, axis2=3)

        cat_hist = self.cat_hist_embed(cat_hist)
        #cat_hist = self.flat(cat_hist)
            
        cat_fut = self.cat_fut_embed(cat_fut)
        #cat_fut = self.flat(cat_fut)
        
        cat_stat = self.cat_stat_embed(cat_stat)
        #cat_stat = self.flat(cat_stat)

        # (batch size / (num time steps * (num historical + future variables) + num static variables) * embedding size)
        hist = self.concat([cont_hist, cat_hist])
        
        return hist, cat_fut, cat_stat


# **From now on, whenever relevant the examples with real data will use the $\xi$ elements created above.**

# The following simplistic model shows how the input layer is used. The goal is to highlight how the data is inputted into a TFT model, by not focusing on its complexity just now.
# 
# First, it takes up the data. Then, in this simplified model it flattens all inputs and uses a dense layer connected to all embeddings at all time points to output the forecast.

# ## Skip connection
# 
# > Adds inputs to layer and then implements layer normalisation
# 
# $$
# \text{LayerNorm}(a + b),
# $$ {#eq-skip}
# 
# for $a$ and $b$ tensors of the same dimension and $\text{LayerNorm}(\cdot)$ being the layer normalisation (@ba2016layer), ie subtracting $\mu^l$ and dividing by $\sigma^l$ defined as:
# 
# $$
# \mu^l = \frac{1}{H} \sum_{i=1}^H n_i^l \quad \sigma^l = \sqrt{\frac{1}{H} \sum_{i=1}^H (n_i^l - \mu^l)^2},
# $$ {#eq-layernorm}
# 
# with $H$ denoting the number of $n$ hidden units in a layer $l$.
# 
# * Adding a layer's inputs to its outputs is also called "skip connection"
# * The layer is then normalised [@ba2016layer] to avoid having the numbers grow too big, which is detrimental for gradient transmission
#   * Layer normalisation uses the same computation both during training and inference times, and is particularly suitable for time series

# ## Gated linear unit (GLU)
# 
# > Linear layer that learns how much to gate vs let pass through
# 
# Using input $\gamma \in \mathbb{R}^{d_{\text{model}}}$ and the subscript $\omega$ to index weights, 
# 
# $$
# \text{GLU}_{\omega}(\gamma) = \sigma(W_{4, \omega} \gamma + b_{4, \omega}) \odot (W_{5, \omega} \gamma + b_{5, \omega}),
# $$ {#eq-GLU}
# 
# where $\mathbf{W} \in \mathbf{R}^{(d_{\text{model}} \times d_{\text{model}})}$ is a matrix of weights and $\mathbf{b} \in \mathbf{R}^{d_{model}}$ is a vector of biases. Importantly, $\mathbf{W}$ and $\mathbf{b}$ are indexed with $_{\omega}$ to denote weight-sharing (within each variable) when the layer is time-distributed.
# 
# @dauphin2017language introduced GLUs. Their intuition is to train two versions of a dense layer in the same data, but one of them having a sigmoid activation (which outputs values between zero and one), then multiplying each hidden unit.
# 
# The result could be zero or very close to zero through the Hadamard multipliciation, which in practice means that the network would not be affected by that data (ie, the data $\gamma$ would be gated out). The first term, with the sigmoid, is the gate that determines what percentage of the linear layer passes through.
# 
# According to @lim2021temporal, GLUs:
# 
# * *"... reduce the vanishing gradient problem for deep architectures by providing a linear path for gradients while retaining non-linear capabilities"* and
# * *"... provide flexibility to suppress any parts of the architecture that are not required for a given dataset"*
# 
# The GLU is a key part of the Gated Residual Network, described in @sec-GRN.

# In[47]:


#| code-fold: hide

class GatedLinearUnit(keras.Layer):
    def __init__(
        self,
        d_model:int=16, # Embedding size, $d_\text{model}$
        dropout_rate:float|None=None, # Dropout rate
        use_time_distributed:bool=True, # Apply the GLU across all time steps?
        activation:str|callable=None, # Activation function
        **kwargs
    ):
        "Gated Linear Unit dynamically gates input data"
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed
        self.activation = activation

    def build(self, input_shape):
        super(GatedLinearUnit, self).build(input_shape)
        self.dropout = layers.Dropout(self.dropout_rate) if self.dropout_rate is not None else None
        self.activation_layer = layers.Dense(self.d_model, activation=self.activation)
        self.gate_layer = layers.Dense(self.d_model, activation='sigmoid')
        self.multiply = layers.Multiply()

        if self.use_time_distributed:
            self.activation_layer = layers.TimeDistributed(self.activation_layer)
            self.gate_layer = layers.TimeDistributed(self.gate_layer)

    def call(
        self, 
        inputs, 
        training=None
    ):
        """List of outputs with shape: [
            (batch size, ..., d_model),
            (batch size, ..., d_model)
        ]"""
        if self.dropout is not None and training:
            inputs = self.dropout(inputs)

        activation_output = self.activation_layer(inputs)
        gate_output = self.gate_layer(inputs)
        return self.multiply([activation_output, gate_output]), gate_output

    def get_config(self):
        config = super(GatingLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
            'use_time_distributed': self.use_time_distributed,
            'activation': self.activation
        })
        return config


# ## Gated residual network (GRN) {#sec-GRN}
# 
# $$
# \text{GRN}_{\omega}(a, c) = \text{LayerNorm}(a + \text{GLU}_{\omega}(W_{1, \omega} \text{ELU}(W_{2, \omega} a + b_{2, \omega} + W_{3, \omega} c) + b_{1, w}))
# $$ {#eq-GRN}
# 
# * Breaking down $\text{GRN}_{\omega}(a, c)$:
#     * *1st step*: $\eta_{2} = \text{ELU}(W_{2, \omega} a + b_{2, \omega} + W_{3, \omega} c)$ (where the additional context $c$ might be zero) as in @eq-dense but adapted for the added context if any and with $\text{ELU}(\cdot)$ as the activation function,
#     * *2nd step*: $\eta_{1} = W_{1, \omega} \eta_{2} + b_{1, w}$ as in @eq-dense,
#     * *3rd step*: $\text{LayerNorm}(a + \text{GLU}_{\omega}(\eta_{1}))$ as in @eq-skip and @eq-GLU
# * $\text{ELU}(\cdot)$ is the Exponential Linear Unit activation function (@clevert2015fast)
#     * Unlike ReLUs, ELUs allow for negative values, which pushes unit activations closer to zero at a lower computation complexity, and producing more accurate results
# * The GRN is a key building block of the TFT
#     * Helps keep information only from relevant input variables
#     * Also keeps the model as simple as possible by only applying non-linearities when relevant
# 
# Note that the GRN can take all types of time series inputs, ie continuous historical, categorical historical and categorical future, but not categorical static data.

# In[52]:


#| code-fold: hide

class GatedResidualNetwork(keras.layers.Layer):
    def __init__(
        self, 
        d_model:int=16, # Embedding size, $d_\text{model}$
        output_size=None, 
        dropout_rate=None, 
        use_time_distributed=True, 
        **kwargs
    ):
        "Gated residual network"
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.d_model = d_model
        self.output_size = output_size if output_size is not None else d_model
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed

    def build(self, input_shape):
        super(GatedResidualNetwork, self).build(input_shape)
        self.dense = layers.Dense(self.output_size)
        self.hidden_dense = layers.Dense(self.d_model)
        self.hidden_activation = layers.Activation('elu')
        self.context_dense = layers.Dense(self.d_model, use_bias=False)
        self.gating_layer = GatedLinearUnit(
            d_model=self.output_size, 
            dropout_rate=self.dropout_rate, 
            use_time_distributed=self.use_time_distributed, 
            activation=None)
        self.add = layers.Add()
        self.l_norm = layers.LayerNormalization()

        if self.use_time_distributed:
            self.dense = layers.TimeDistributed(self.dense)
            self.hidden_dense = layers.TimeDistributed(self.hidden_dense)
            self.context_dense = layers.TimeDistributed(self.context_dense)

    def call(self, inputs, additional_context=None, training=None):
        # Setup skip connection
        skip = self.dense(inputs) if self.output_size != self.d_model else inputs
        
        # 1st step: eta2
        hidden = self.hidden_dense(inputs)

        # Context handling
        if additional_context is not None:
            hidden += self.context_dense(additional_context)

        hidden = self.hidden_activation(hidden)

        # 2nd step: eta1 and 3rd step
        gating_layer, gate = self.gating_layer(hidden)
        
        # Final step
        GRN = self.add([skip, gating_layer])
        GRN = self.l_norm(GRN)

        return GRN, gate

    def get_config(self):
        config = super(GatedResidualNetwork, self).get_config()
        config.update({
            'd_model': self.d_model,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'use_time_distributed': self.use_time_distributed
        })
        return config


# The example below uses transformed data $\xi_t^{(j)}$ for $j=0$ as an example.

# ## Variable selection networks
# 
# $$
# \sum_{j=1}^{m_{\chi}} \upsilon_{\chi_t}^{(j)} \tilde{\xi}_t^{(j)},
# $$ {#eq-VSN}
# 
# with $j$ indexing the input variable and $m$ being the number of features, $\upsilon_{\chi_t}^{(j)}$ standing for variable $j$'s selection weight, and $\tilde{\xi}_t^{(j)}$ defined as:
# 
# $$
# \tilde{\xi}_t^{(j)} = \text{GRN}(\xi_t^{(j)}).
# $$ {#eq-embed}
# 
# * In the paper, they are represented in the bottom right of Fig. 2
# * Note there are separate variable selection networks for different input groups:
#   * `static_variable_selection`
#     * does not have static context as input, it already *is* the static information
#   * `temporal_variable_selection`
#     * used for both historical and known future inputs
#     * includes static contexts
# * Both of these functions take the result of the transformed data, ie embeddings for categorical variables and a linear layer for continuous variables
#   * static variables are always categorical
#   * temporal variables can be either categorical or continuous
#   * in any case, following @lim2021temporal, the resulting transformation is expected to have the same dimension as `d_model`

# In[57]:


class StaticVariableSelection(keras.Layer):
    def __init__(
        self, 
        d_model:int=16, # Embedding size, $d_\text{model}$
        dropout_rate:float=0., 
        **kwargs
    ):
        "Static variable selection network"
        super(StaticVariableSelection, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # Define GRNs for the transformed embeddings
        self.grns_transformed_embeddings = []  # This will be a list of GRN layers

        self.flat = layers.Flatten()
        self.softmax = layers.Activation('softmax')
        self.mult = layers.Multiply()

    def build(self, input_shape):
        super(StaticVariableSelection, self).build(input_shape)
        
        num_static = input_shape[2]

        # Define the GRN for the sparse weights
        self.grn_sparse_weights = GatedResidualNetwork(
            d_model=self.d_model,
            output_size=num_static,
            use_time_distributed=False
        )

        for i in range(num_static):
            # Create a GRN for each static variable
            self.grns_transformed_embeddings.append(
                GatedResidualNetwork(self.d_model, use_time_distributed=False)
            )

    def call(self, inputs, training=None):
        _, _, num_static, _ = inputs.shape # batch size / one time step (since it's static) / num static variables / d_model

        flattened = self.flat(inputs)

        # Compute sparse weights
        grn_outputs, _ = self.grn_sparse_weights(flattened, training=training)
        sparse_weights = self.softmax(grn_outputs)
        sparse_weights = keras.ops.expand_dims(sparse_weights, axis=-1)

        # Compute transformed embeddings
        transformed_embeddings = []
        for i in range(num_static):
            embed, _ = self.grns_transformed_embeddings[i](inputs[:, 0, i:i+1, :], training=training)
            transformed_embeddings.append(embed)
        transformed_embedding = keras.ops.concatenate(transformed_embeddings, axis=1)

        # Combine with sparse weights
        combined = self.mult([sparse_weights, transformed_embedding])
        static_vec = keras.ops.sum(combined, axis=1)

        return static_vec, sparse_weights

    def get_config(self):
        config = super(StaticVariableSelectionLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate
        })
        return config


# In[61]:


#| code-fold: hide

class TemporalVariableSelection(keras.Layer):
    def __init__(
        self, 
        d_model:int=16, # Embedding size, $d_\text{model}$
        dropout_rate:float=0., 
        **kwargs
    ):
        "Temporal variable selection"
        super(TemporalVariableSelection, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        self.mult = layers.Multiply()

    def build(self, input_shape):
        super(TemporalVariableSelection, self).build(input_shape)
        self.batch_size, self.time_steps, self.num_input_vars, self.d_model = input_shape[0]

        self.var_sel_weights = GatedResidualNetwork(
            d_model=self.d_model,
            output_size=self.num_input_vars,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
        )
        self.softmax = layers.Activation('softmax')
    
        # Create a GRN for each temporal variable
        self.grns_transformed_embeddings = GatedResidualNetwork(
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True
        )

    def call(
        self, 
        inputs, # List of temporal embeddings, static context
        training=None
    ):
        temporal_embeddings, static_context = inputs

        flattened_embed = keras.ops.reshape(
            temporal_embeddings,
            [-1, self.time_steps, self.num_input_vars * self.d_model]
        )
        parallel_variables = keras.ops.reshape(
            temporal_embeddings, 
            [-1, self.time_steps, self.d_model]
        ) # tensor is shaped this way so that a GRN can be applied to each variable of each batch
        c_s = keras.ops.expand_dims(static_context, axis=1)

        # variable weights
        grn_outputs, _ = self.var_sel_weights(flattened_embed, c_s, training=training)
        variable_weights = self.softmax(grn_outputs)
        variable_weights = keras.ops.expand_dims(variable_weights, axis=2)

        # variable combination
        # transformed_embeddings = [
        #     grn_layer(temporal_embeddings[:, :, i, :], training=training)[0]
        #     for i, grn_layer in enumerate(self.grns_transformed_embeddings)
        # ]
        transformed_embeddings, _ = self.grns_transformed_embeddings(parallel_variables, training=training)
        transformed_embeddings = keras.ops.reshape(
            transformed_embeddings,
            [-1, self.time_steps, self.num_input_vars, self.d_model]
        )
        #transformed_embeddings = keras.ops.stack(transformed_embeddings, axis=2)
        temporal_vec = keras.ops.einsum('btij,btjk->btk', variable_weights, transformed_embeddings)
        return temporal_vec, keras.ops.squeeze(variable_weights, axis=2)


# The main input to `temporal_variable_selection` are the transformed *temporal* variables $\xi_t^{(j)}$.
# 
# The selection of the temporal variables also requires the context from the static variables $c_s$, created below by passing `static_vars`, the output from the static variable selection unit (see @eq-static_var_sel), into a GRN.

# ## Sequence-to-sequence layer (LSTM)

# This  the `TemporalFeatures` layer implements the following transformation:
# 
# $$ \text{LSTM} :
# \tilde{\xi}_{t-k:t}, \tilde{\xi}_{t:\tau_{\text{max}}} \in \mathbb{R}^{k + \tau_{\text{max}} X d_{\text{model}}} \to \phi(t, n) \in \mathbb{R}^{k + \tau_{\text{max}} X d_{\text{model}}}, n \in [-k, \tau_{\text{max}}],
# $$ {#eq-seqtoseq}
# 
# where the starting cell and hidden states of $\text{LSTM}_{t-k:t}$ are $c_c$ and $c_h$, each calculated as $c_p = GRN(\xi^{(j)}), p \in (c, h)$ and $j$ denoting static variables.
# 
# Finally, the `TemporalFeatures` layer compares the input data $\tilde{\xi}_{t-k:\tau_{\text{max}}}$ with the non-linear transformation $\phi(t, n)$, as follows:
# 
# $$
# \tilde{\phi}(t, n) = \text{LayerNorm}(\tilde{\xi}_{t+n} + \text{GLU}_{\tilde{\phi}}(\phi(t, n))).
# $$

# In[65]:


class TemporalFeatures(keras.Layer):
    def __init__(
        self, 
        d_model:int=16, # Embedding size, $d_\text{model}$
        dropout_rate:float=0., # Dropout rate
        **kwargs
    ):
        super(TemporalFeatures, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(TemporalFeatures, self).build(input_shape)
        self.hist_encoder = layers.LSTM(
            units=self.d_model,
            return_sequences=True,
            return_state=True,
            stateful=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=self.dropout_rate,
            unroll=False,
            use_bias=True
        )
        self.fut_decoder = layers.LSTM(
            units=self.d_model,
            return_sequences=True,
            return_state=False,
            stateful=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=self.dropout_rate,
            unroll=False,
            use_bias=True
        )
        self.lstm_glu = GatedLinearUnit(
            d_model=self.d_model, # Dimension of the GLU
            dropout_rate=self.dropout_rate, # Dropout rate
            use_time_distributed=True, # Apply the GLU across all time steps?
            activation=None # Activation function
        )
        self.add = layers.Add()
        self.l_norm = layers.LayerNormalization()

    def call(self, inputs, training=None):
        historical_features, future_features, c_h, c_c = inputs
        input_embeddings = keras.ops.concatenate(
            [historical_features, future_features],
            axis=1
        )

        history_lstm_encoder, state_h, state_c = self.hist_encoder(
            historical_features,
            initial_state=[
                c_h, # short-term state
                c_c  # long-term state
            ],
            training=training
        )

        future_lstm_decoder = self.fut_decoder(
            future_features,
            initial_state=[
                state_h, # short-term state
                state_c  # long-term state
            ],
            training=training
        )
        
        # this step concatenates at the time dimension, ie
        # the time series of history internal states are now
        # concated in sequence with the series of future internal states
        # $\phi(t,n) \in \{\phi(t,-k), \dots, \phi(t, \tau_{\text{max}})\}$
        lstm_layer = keras.ops.concatenate([history_lstm_encoder, future_lstm_decoder], axis=1)
        
        # Apply gated skip connection
        lstm_layer, _ = self.lstm_glu(
            lstm_layer,
            training=training
        )
        outputs = self.add([lstm_layer, input_embeddings])
        outputs = self.l_norm(outputs)
        # it's the temporal feature layer that is fed into the Temporal Fusion Decoder
        # its dimensions are (batch size / num time steps historical + future / hidden size)

        return outputs


# The graph below plots the temporal futures resulting from the function above, using the first batch of data as an example. Each curve is the time series of one element of the embedding vector, which in turn contains all the relevant information from the time-varying inputs, both categorical and continuous, after being filtered by the network. The black vertical line marks the point at which the temporal futures is relying on future known information.
# 
# Note that even with these randomly initiated LSTM layers, it is already possible to see the obvious fact that the information content from historical input (left to the vertical line) is different compared to the known future data (ie, information from the dates; to the right of the vertical line).
# 
# Still, the future part has *some* information, which might be useful in nowcasting or predicting inflation farther out.

# ## Static enrichment

# This step is responsible for adding static information on the country for which inflation is being nowcasted to the temporal features.
# 
# This is achieved by a GRN layer as follows:
# 
# $$
# \theta(t, n) = \text{GRN}_{\theta}(\tilde{\phi}(t, n), c_e)
# $$

# ## Attention components

# * Attention mechanisms use relationships between keys $K \in \mathbf{R}^{N \times d_{attention}}$ and queries $Q \in \mathbf{R}^{N \times d_{attention}}$ to scale a vector of values $V \in \mathbf{R}^{N \times d_V}$: $\text{Attention}(Q, K, V) = A(Q, K) V$
#     * $N$ is the number of timesteps going into the attention layer (the number of lags $k$ plus the number of periods to be forecasted $\tau_{\text{max}}$)
#     * $A(\cdot)$ is a normalisation function
#         * After @vaswani2017attention, the canonical choice for $A(\cdot)$ is the scaled dot-product: $A(Q, K) = \text{Softmax}(\frac{Q K^{T}}{\sqrt{d_{attention}}} )$
#     
# * The TFT uses a modified attention head to enhance the explainability of the model
# * Specifically, the transformer block (multi-head attention) is modified to:
#     * share values in each head, and
#     * employ additive aggregation of all heads
# * More formally, compare the interpretable multi-head attention (used in this paper) with the canonical multi-head attention:
#     * $\text{InterpretableMultiHead}(Q, K, V) = \tilde{H} W_{H}$, with:
#         * $\begin{aligned}\tilde{H} &= \tilde{A}(Q, K) V W_V \\
#         &= \{\frac{1}{m_H} \sum^{m_{H}}_{h=1} A(Q W^{(h)}_Q, K W^{(h)}_K) \} V W_V \\
#         &= \frac{1}{m_H} \sum^{m_{H}}_{h=1} \text{Attention}(Q W^{(h)}_Q, K W^{(h)}_K, V W_V)
#         \end{aligned}$
#     * $\text{MultiHead}(Q, K, V) = [H_1, \dots, H_{m_H}] W_H$, with:
#         * $H_h = \text{Attention}(Q W^{(h)}_Q, K W^{(h)}_K, V W_V^{(h)}) $

# ### Decoder mask for self-attention layer

# In[69]:


def get_decoder_mask(
    self_attention_inputs # Inputs to the self-attention layer
):
    "Determines shape of decoder mask"
    len_s = keras.ops.shape(self_attention_inputs)[1] # length of inputs
    bs = keras.ops.shape(self_attention_inputs)[0] # batch shape
    mask = keras.ops.cumsum(keras.ops.eye(len_s), axis=0)

    ### warning: I had to manually implement some batch-wise shape here 
    ### because the new keras `eye` function does not have a batch_size arg.
    ### inspired by: https://github.com/tensorflow/tensorflow/blob/v2.14.0/tensorflow/python/ops/linalg_ops_impl.py#L30
    ### <hack>
    mask = keras.ops.expand_dims(mask, axis=0)    
    mask = keras.ops.tile(mask, (bs, 1, 1))
    ### </hack>

    return mask


# Note that it produces an upper-triangular matrix of ones:

# ## Scaled dot product attention layer
# 
# * This is the same as Eq. (1) of @vaswani2017attention 
#     * except that in this case the dimension of the value vector is the same $d_{\text{attn}} = d_{\text{model}} / m_{\text{Heads}}$ as for the query and key vectors
# * As discussed in the paper, additive attention outperforms dot product attention for larger $d_{\text{model}}$ values, so the attention is scaled back to smaller values

# In[73]:


class ScaledDotProductAttention(keras.Layer):
    def __init__(
        self,
        dropout_rate:float=0.0, # Will be ignored if `training=False`
        **kwargs
    ):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(ScaledDotProductAttention, self).build(input_shape)
        self.dropout = layers.Dropout(rate=self.dropout_rate)
        self.activation = keras.layers.Activation('softmax')
        self.dot_22 = layers.Dot(axes=(2, 2))
        self.dot_21 = layers.Dot(axes=(2, 1))
        self.lambda_layer = layers.Lambda(lambda x: (-1e9) * (1. - keras.ops.cast(x, 'float32')))
        self.add = layers.Add()

    def call(
        self,
        q, # Queries, tensor of shape (?, time, D_model)
        k, # Keys, tensor of shape (?, time, D_model)
        v, # Values, tensor of shape (?, time, D_model)
        mask, # Masking if required (sets Softmax to very large value), tensor of shape (?, time, time)
        training=None, # Whether the layer is being trained or used in inference
    ):
        # returns Tuple (layer outputs, attention weights)
        scale = keras.ops.sqrt(keras.ops.cast(keras.ops.shape(k)[-1], dtype='float32'))
        attention = self.dot_22([q, k]) / scale
        #attention = keras.ops.einsum("bij,bjk->bik", q, keras.ops.transpose(k, axes=(0, 2, 1))) / scale
        if mask is not None:
            mmask = self.lambda_layer(mask)
            attention = self.add([attention, mmask])
        attention = self.activation(attention)
        if training:
            attention = self.dropout(attention)
        output = self.dot_21([attention, v])
        #output = keras.ops.einsum("btt,btd->bt", attention, v)
        return output, attention


# Testing without masking:

# ... and with masking:

# ## Softmax

# A small detour to illustrate the softmax function. 
# 
# The $i^{\text{th}}$ element of $\text{Softmax}(x)$, with $x \in \mathbf{R}^K$ is:
# 
# $$
# \text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}}
# $$

# For example, see the values below for an input vector $x$ ($K=5$ in this example):

# As can be seen above, the softmax function really makes the largest numbers stand out from the rest.
# 
# Note also that $-\infty$ results in 0.

# ## Interpretable Multi-head attention

# * When values are shared in each head and then are aggregated additively, each head can still learn different temporal patterns (from their own unique queries and keys), but with the same input values.
#     * In other words, they can be interpreted as an ensemble over the attention weights
#     * the paper doesn't mention this explicitly, but the ensemble is equally-weighted - maybe there is some performance to be gained by having some way to weight the different attention heads 🤔, such as having a linear layer combining them... will explore in the future

# In[80]:


class InterpretableMultiHeadAttention(keras.Layer):
    def __init__(
        self,
        n_head:int,
        d_model:int, # Embedding size, $d_\text{model}$
        dropout_rate:float, # Will be ignored if `training=False`
        **kwargs
    ):
        super(InterpretableMultiHeadAttention, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_qk = self.d_v = d_model // n_head # the original model divides by number of heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(InterpretableMultiHeadAttention, self).build(input_shape)
        
        # using the same value layer facilitates interpretability
        vs_layer = layers.Dense(self.d_v, use_bias=False, name="shared_value")

        # creates list of queries, keys and values across heads
        self.qs_layers = [layers.Dense(self.d_qk) for _ in range(self.n_head)]
        self.ks_layers = [layers.Dense(self.d_qk) for _ in range(self.n_head)]
        self.vs_layers = [vs_layer for _ in range(self.n_head)]

        self.attention = ScaledDotProductAttention(dropout_rate=self.dropout_rate)
        self.w_o = layers.Dense(self.d_v, use_bias=False, name="W_v") # W_v in Eqs. (14)-(16), output weight matrix to project internal state to the original TFT
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(
        self,
        q, # Queries, tensor of shape (?, time, d_model)
        k, # Keys, tensor of shape (?, time, d_model)
        v, # Values, tensor of shape (?, time, d_model)
        mask=None, # Masking if required (sets Softmax to very large value), tensor of shape (?, time, time)
        training=None
    ):
        heads = []
        attns = []
        for i in range(self.n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](q)
            vs = self.vs_layers[i](v)
           
            head, attn = self.attention(qs, ks, vs, mask, training=training)
            if training:
                head = self.dropout(head)
            heads.append(head)
            attns.append(attn)
        head = keras.ops.stack(heads) if self.n_head > 1 else heads[0]

        outputs = keras.ops.mean(head, axis=0) if self.n_head > 1 else head # H_tilde
        outputs = self.w_o(outputs)
        if training:
            outputs = self.dropout(outputs)

        return outputs, attn


# #### Example usage

# ## Complete TFT model

# In[85]:


@keras.saving.register_keras_serializable() # Make sure custom class can be saved with model.save()
class TFT(keras.Model):
    def __init__(
        self,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        d_model:int=16, # Embedding size, $d_\text{model}$
        output_size:int=1, # How many periods to nowcast/forecast?
        n_head:int=4,
        dropout_rate:float=0.1,
        **kwargs
    ):
        super(TFT, self).__init__(**kwargs)
        self.quantiles = quantiles
        self.d_model = d_model
        self.output_size = output_size
        self.n_head = n_head
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(TFT, self).build(input_shape)
        
        self.input_layer = InputTFT(
            d_model=self.d_model,
            name="input"
        )
        self.svars = StaticVariableSelection(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name="static_variable_selection"
        )
        self.tvars_hist = TemporalVariableSelection(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name="historical_variable_selection"
        )
        self.tvars_fut = TemporalVariableSelection(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name="future_variable_selection"
        )
        self.static_context_s_grn = GatedResidualNetwork(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            name="static_context_for_variable_selection"
        )
        self.static_context_h_grn = GatedResidualNetwork(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            name="static_context_for_LSTM_state_h"
        )
        self.static_context_c_grn = GatedResidualNetwork(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            name="static_context_for_LSTM_state_c"
        )
        self.static_context_e_grn = GatedResidualNetwork(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            name="static_context_for_enrichment_of"
        )
        self.temporal_features = TemporalFeatures(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name="LSTM_encoder"
        )
        self.static_context_enrichment = GatedResidualNetwork(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            name="static_context_enrichment"
        )
        self.attention = InterpretableMultiHeadAttention(
            n_head=4,
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name="attention_heads"
        )
        self.attention_gating = GatedLinearUnit(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            activation=None,
            name="attention_gating"
        )
        self.attn_grn = GatedResidualNetwork(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            name="output_nonlinear_processing"
        )
        self.final_skip = GatedLinearUnit(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            activation=None,
            name="final_skip_connection"
        )
        self.add = layers.Add()
        self.l_norm = layers.LayerNormalization()
        
        self.flat = layers.Flatten(name="flatten")


        # Output layers:
        # In order to enforce monotoncity of the quantiles forecast only the lowest quantile
        # from a base forecast layer, and use output_len - 1 additional layers with ReLU activation
        # to produce the difference between the current quantile and the previous one
        output_len = len(self.quantiles)

        self.base_output_layer = layers.TimeDistributed(
            layers.Dense(1),
            name="output"
        )
        def elu_plus(x):
            return keras.activations.elu(x) + 1

        self.quantile_diff_layers = [
            layers.TimeDistributed(
                layers.Dense(1, activation=elu_plus),
                name=f"quantile_diff_{i}"
            ) 
            for i in range(output_len - 1)
        ]
    
    def call(
        self,
        inputs,
        training=None
    ):
        "Creates the model architecture"
        
        # embedding the inputs
        cont_hist, cat_hist, cat_fut, cat_stat = inputs
        if len(cat_stat.shape) == 2:
            cat_stat = keras.ops.expand_dims(cat_stat, axis=-1)
            
        xi_hist, xi_fut, xi_stat = self.input_layer([cont_hist, cat_hist, cat_fut, cat_stat])

        # selecing the static covariates
        static_selected_vars, static_selection_weights = self.svars(xi_stat, training=training)

        # create context vectors from static data
        c_s, _ = self.static_context_s_grn(static_selected_vars, training=training) # for variable selection
        c_h, _ = self.static_context_h_grn(static_selected_vars, training=training) # for LSTM state h
        c_c, _ = self.static_context_c_grn(static_selected_vars, training=training) # for LSTM state c
        c_e, _ = self.static_context_e_grn(static_selected_vars, training=training) # for context enrichment of post-LSTM features

        # temporal variable selection
        hist_selected_vars, hist_selection_weights = self.tvars_hist(
            [xi_hist, c_s],
            training=training
        )
        fut_selected_vars, fut_selection_weights = self.tvars_fut(
            [xi_fut, c_s],
            training=training
        )
        input_embeddings = keras.ops.concatenate(
            [hist_selected_vars, fut_selected_vars],
            axis=1
        )

        features = self.temporal_features(
            [hist_selected_vars, fut_selected_vars, c_h, c_c],
            training=training
        )
        
        # static context enrichment
        enriched, _ = self.static_context_enrichment(
            features, 
            additional_context=keras.ops.expand_dims(c_e, axis=1),
            training=training
        )
        mask = get_decoder_mask(enriched)
        attn_output, self_attn = self.attention(
            q=enriched,
            k=enriched,
            v=enriched,
            mask=mask,
            training=training
        )
        attn_output, _ = self.attention_gating(attn_output)
        output = self.add([enriched, attn_output])
        output = self.l_norm(output)
        output, _ = self.attn_grn(output)
        output, _ = self.final_skip(output)
        output = self.add([features, output])
        output = self.l_norm(output)
        
        # Base quantile output
        base_output = output[Ellipsis,hist_selected_vars.shape[1]:,:]
        base_quantile = self.base_output_layer(base_output)
                
        # Additional layers for remaining quantiles
        quantile_outputs = [base_quantile]
        for i in range(len(self.quantiles) - 1):
            quantile_diff = self.quantile_diff_layers[i](base_output)
            quantile_output = quantile_outputs[-1] + quantile_diff
            quantile_outputs.append(quantile_output)

        final_output = keras.ops.concatenate(quantile_outputs, axis=-1)
        
        return final_output


# In[89]:


class NowcastingData(Dataset):
    def __init__(
        self,
        n_samples: int,
        df_daily_input: pd.DataFrame,
        df_target: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        **kwargs
    ):
        self.n_samples = n_samples
        self.df_daily_input = df_daily_input.loc[start_date:end_date]
        self.df_target = df_target
        self.kwargs = kwargs

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X, y = sample_nowcasting_data(
            df_daily_input=self.df_daily_input,
            df_target=self.df_target,
            **self.kwargs

        )
        return X, y


# In[ ]:


quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
@keras.saving.register_keras_serializable() # Make sure model with custom loss function can be saved with model.save()
def quantile_loss(y_true, y_pred):
    # Assuming quantiles is a numpy array of shape (q,)
    # Extend the shape of y_true to (b, t, 1) to align with y_pred's shape (b, t, q)
    y_true_extended = keras.ops.expand_dims(y_true, axis=-1)

    # Compute the difference in a broadcasted manner
    pred_diff = y_true_extended - y_pred

    # Calculate the quantile loss using broadcasting
    # No need for a loop; numpy will broadcast quantiles across the last dimension
    
    q = keras.ops.array(quantiles)
    q_loss = keras.ops.maximum(q * pred_diff, (q - 1) * pred_diff)
    
    # Average over the time axis
    q_loss = keras.ops.mean(q_loss, axis=-2)

    # Sum over the quantile axis to get the final loss
    final_loss = keras.ops.sum(q_loss, axis=-1)

    return final_loss


# # Checking the prediction for the testing data

# ## Including speech data
# For now, simply merge the speech embeddings with the daily DataFrame

# ___

# # Using the TFT model

# ## Data
# 
# In this example, we will use a simple inflation panel dataset.

# Clean the titles from the metadata

# ### Data preparation
# 
# This crucial step involves:
# * measuring the mean and standard deviation of the inflation of each country in the training dataset
# * using the values above to standardise the training, validation and testing datasets

# ## A simple dense layer
# 
# This first model is autoregressive: it takes in $p$ lags of an inflation series $\pi_i$ (in other words, $\pi_{i, t-p}, ..., \pi_{i, t-1}$) to predict the period $\pi_{i,t}$.
# 
# Note that the model is very simple:
# * each country's inflation series is only predicted by its past values
# * the fully connected linear layer learns to pick up any meaningful non-linear interactions between lags, but there is no intrinsic meaning in the order of the lags
# * this network will always take in as input a $p$-sized vector of lagged data

# ### Data formatting

# Let's create a simple function that will take a data frame and return a (input, output) tuple for the model.

# ### Model

# ## LSTM neural network

# ## LSTM with other countries' inflation data

# The difference between this and the previous one is that it is not only autoregressive, but also considers past data from other countries.

# ### Data formatting

# ### Model

# ## Model with date features
# 
# Repeating date features (eg, day in the week, month, quarter and year, week in the month, quarter and year, month in the quarter and year, and quarter in year) can be embedded and included in the model.

# ## Creating a TFT model

# # Test

# # References {.unnumbered}

# 
