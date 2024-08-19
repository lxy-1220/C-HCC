import joblib
import streamlit as st
import plotly.express as px
from pysurvival.utils import load_model
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict_survival(self, X, times):
        return self.model.predict_survival(X, times)

if 'NMTLR' not in st.session_state:
    st.session_state["RSF"] = load_model("RSF.zip")
    st.session_state["NMTLR"] = joblib.load("NMTLR_model.joblib")


if 'patients' not in st.session_state:
    st.session_state['patients'] = []
if 'display' not in st.session_state:
    st.session_state['display'] = 1

def set_background():
    page_bg_img = '''
    <style>
    .st-emotion-cache-7xfda7 {line-height: 3.6}
    .st-emotion-cache-z5fcl4 {padding: 1rem 1rem 2rem;width:92%}
    h1 {background-color: #B266FF;border-left: red solid 0.6vh}
    .st-emotion-cache-ocqkz7 {width:80%;margin-left:10%}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background()


def get_select1():
    dic = {
        "Gender": ["Female", "Male"],
        "Race": ["American Indian/Alaska Native", "Asian or Pacific Islander", "Black", "White"],
        "Marital_status": ["Married", "Other"],
        "Histological_type": ["8170", "8172", "8173", "8174", "8175"],
        "Grade": ["Undifferentiated; anaplastic; Grade IV", "Moderately differentiated; Grade II", 
                  "Poorly differentiated; Grade III", "Well differentiated; Grade I"],
    }
    return dic

 
def get_select2():
    dic = {
        "N": ["N1", "N2"],
        "M": ["M1", "M2"],
        "Number_of_tumors": ["1", ">1"],
        "Surgery": ["Lobectomy", "Local tumor destruction", "No", "Wedge or segmental resection"],

    }
    return dic


def get_select3():
    dic = {
        "Radiotherapy": ["No", "Yes"],
        "AJCC_Stage": ["I", "II", "III", "IV"],
        "T": ["T1", "T2", "T3", "T4"],
        "AFP": ["Negative/normal; within normal limits", "Positive/elevated"],
        "Chemotherapy": ["No/Unknown", "Yes"]
    }
    return dic


def plot_below_header():
    # col1, col2 = st.columns([1, 9])
    st.session_state['display'] = ['Single', 'Multiple'].index(
        st.radio("Display", ('Single', 'Multiple'), st.session_state['display'], horizontal=True))
    plot_survival()
    col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 2, 2])
    # with col1:
    #     st.write('')
    #     st.write('')
    #     st.write('')
    #     st.write('')
    #     st.write('')
    #     st.write('')
    #     st.write('')
    #     st.write('')
        # st.session_state[''] = ['Single', 'Multiple'].index(
        #     st.radio("", ('Single', 'Multiple'), st.session_state['']))
        # st.radio("Model", ('DeepSurv', 'NMTLR','RSF','CoxPH'), 0,key='model',on_change=predict())
    # with col2:
    #     plot_survival()
    with col4:
        st.metric(
            label='1-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['1-year'] * 100)
        )
    with col5:
        st.metric(
            label='3-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['3-year'] * 100)
        )
    with col6:
        st.metric(
            label='5-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['5-year'] * 100)
        )
    st.write('')
    st.write('')
    st.write('')
    plot_patients()


def plot_survival():
    pd_data = pd.concat(
        [
            pd.DataFrame(
                {
                    'Survival': item['survival'],
                    'Time': item['times'],
                    'Patients': [item['No'] for i in item['times']]
                }
            ) for item in st.session_state['patients']
        ]
    )
    if st.session_state['display']:
        fig = px.line(pd_data, x="Time", y="Survival", color='Patients', range_y=[0, 1])
    else:
        fig = px.line(pd_data.loc[pd_data['Patients'] == pd_data['Patients'].to_list()[-1], :], x="Time", y="Survival",
                      range_y=[0, 1])
    fig.update_layout(title={
        'text': 'Estimated Survival Probability',
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(
            size=25
        )
    },
                      # plot_bgcolor="LightGrey",
                      xaxis_title="Time, month",
                      yaxis_title="Survival probability",
                      template = "plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_patients():
    patients = pd.concat(
        [
            pd.DataFrame(
                dict(
                    {
                        'Patients': [item['No']],
                        'Model': [item["use_model"]],
                        '1-Year': ["{:.2f}%".format(item['1-year'] * 100)],
                        '3-Year': ["{:.2f}%".format(item['3-year'] * 100)],
                        '5-Year': ["{:.2f}%".format(item['5-year'] * 100)]
                    },
                    **item['arg']
                )
            ) for item in st.session_state['patients']
        ]
    ).reset_index(drop=True)
    st.dataframe(patients, hide_index=True)
    col11, col12 = st.columns([8, 2])
    with col12:
        st.download_button(
            label="Download result as csv",
            data=patients.to_csv().encode('utf-8'),
            file_name='result.csv',
            mime='text/csv',
            use_container_width=True
        )


def predict():
    model = st.session_state[st.session_state["model"]]
    input_keys = ['Age', 'Gender', 'Race', 'Marital_status', 'Histological_type', 'Grade', 'AJCC_Stage', 'T',
                  'N', 'M', 'AFP', 'Number_of_tumors', 'Tumor_size', 'Surgery', 'Radiotherapy', 'Chemotherapy']
    all_dic = dict(get_select1(), **get_select2(), **get_select3(), **{"Age": st.session_state["Age"], "Tumor_size": st.session_state["Tumor_size"]})
    test_df = []
    for _ in input_keys:
        if _ in ["Age", "Tumor_size"]:
            test_df.append(all_dic[_]) 
        else:
            test_df.append(all_dic[_].index(st.session_state[_]))
    # # test
    # test_df = [61, 0, 0, 0, 0, 0, 3, 2, 0, 0, 1, 0, 35, 1, 0, 1]
    if st.session_state["model"] == 'NMTLR':
        survival = [model.predict_survival(test_df, _)[0] for _ in range(1, 61)]
    else:
        survival = model.predict_survival(test_df)[0]
    data = { 
        'survival': survival,
        'times': [i for i in range(1, len(survival) + 1)],
        'No': len(st.session_state['patients']) + 1,
        'arg': {key: st.session_state[key] for key in input_keys},
        'use_model': st.session_state["model"],
        '1-year': model.predict_survival(test_df, 12)[0],
        '3-year': model.predict_survival(test_df, 36)[0],
        '5-year': survival[-1],
    }
    st.session_state['patients'].append(
        data
    )
    print('update patients ... ##########')


if 'model' not in st.session_state:
    st.session_state["model"] = "RSF"
st.markdown("<h1 style='text-align: center'>{} Model in Predicting Cancer-Specific Survival of Hepatocellular Carcinoma in Cirrhotic Patients</h1>".format(st.session_state["model"]), unsafe_allow_html=True)
for i in range(2):
    st.write("")
with st.container():
    col1, col2, col3 = st.columns([3, 3, 3])
    with col1:
        st.number_input("Age (years)", min_value=25, max_value=90, value=61, key="Age")
        for _ in get_select1():
            st.selectbox(_, get_select1()[_], index=None, key=_)
    with col2:
        st.number_input("Tumor size (mm)", min_value=4, max_value=461, value=35, key="Tumor_size")
        for _ in get_select2():
            st.selectbox(_, get_select2()[_], index=None, key=_)
        st.selectbox("Please select model 👇", ["RSF", "NMTLR"], key='model')
    with col3:

        for _ in get_select3():
            st.selectbox(_, get_select3()[_], index=None, key=_)
        prediction = st.button(
            'Predict',
            type='primary',
            on_click=predict,
            use_container_width=True
        )

if st.session_state['patients']:
    plot_below_header()

