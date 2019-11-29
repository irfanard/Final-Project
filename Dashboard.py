import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
# from src.view import renderIsiTabBar
from dash.dependencies import Input, Output, State
import pickle
log_model = pickle.load(open('personalloan_predictor.sav','rb'))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_csv('thera_bank.csv')
# trace1 = go.Bar(
#     x=['CD Account'],
#     y=['Personal Loan'] == 0,
#     name='Personal Loan:0'
# )
# trace2 = go.Bar(
#     x=['CD Account'],
#     y=['Personal Loan'] ==1,
#     name='Personal Loan:1'
# )

# data = [trace1, trace2]
# layout = go.Layout(
#     barmode='group'
# )

# fig = go.Figure(data=df, layout=layout)




def generate_table(dataframe, page_size = 10):
     return dash_table.DataTable(
                    id = 'dataTable',
                    columns = [{"name": i, "id": i} for i in dataframe.columns],
                    data=dataframe.to_dict('records'),
                    page_action="native",
                    page_current= 0,
                    page_size= page_size,
                )

app.layout = html.Div([
        html.H1(children='Data Science Final Project'),
        html.P('created by: Irfan Ardiansyah'),
        html.Div([html.Div(children =[
        dcc.Tabs(value = 'tabs', id = 'tabs-1', children = [
            dcc.Tab(value = 'Tabel', label = 'DataFrame Table', children =[
                html.Center(html.H1('DATAFRAME BANK LOAN FOR CUSTOMER CLASSIFICATION')),
                html.Div(children =[
                    html.Div(children =[
                        html.P('Personal Loan:'),
                        dcc.Dropdown(value = '', id='filter-personal_loan', options = [{'label': '0', 'value': 0},
                                                                                      {'label': '1', 'value': 1},
                                                                                      {'label': 'All', 'value': ''}])
                    ], className = 'col-3')
                ], className = 'row'),
                html.Div(children = [
                    html.Div(children =[
                    html.P('Max Rows : '),
                    dcc.Input(
                        id='filter-row',
                        type='number',
                        value=10,
                    )
                ], className = 'col-9')
                ], className = 'row'), 
                html.Br(),
                html.Div(children =[
                        html.Button('search',id = 'filter')
                    ],className = 'col-4'),
                html.Br(),    
                html.Div(id = 'div-table', children =[generate_table(df)
                ])
            ]),

            dcc.Tab(value='Chart-Visual', label='Features Visualization', children =[
                html.Div(children = [
                    html.Div(children = [
                        html.Div([
                            dcc.Graph(
                                id='income-chart',
                                figure={
                                    'data': [{
                                    'x': df['Personal Loan'],
                                    'y': df['Income'],
                                    'type': 'violin',
                                    'name' :'Personal Loan'
                                    }],
                                    'layout':dict(
                                        {'title':'Income - Personal Loan '},
                                        xaxis={'title': 'Personal Loan'},
                                        yaxis={'title':'Count'}
                                    )
                                }
                            )
                        ], className = 'col-6'),

                        html.Div([
                            dcc.Graph(
                                id='ccavg-chart',
                                figure={
                                    'data': [{
                                    'x': df['Personal Loan'],
                                    'y': df['CCAvg'],
                                    'type': 'box',
                                    'name' :'Personal Loan'
                                    }],
                                    'layout':dict(
                                        {'title':'CCAvg - Personal Loan '},
                                        xaxis={'title': 'Personal Loan'}
                                    )
                                }
                            )
                        ], className = 'col-6')
                    ], className = 'row'),

                    # Row 2
                    html.Div(children = [
                        html.Div([
                            dcc.Graph(
                                id='Education-chart',
                                figure={
                                    'data': [
                                        {'x': [1,2,3],'y':[2003,1221,1296],'type':'bar','name':'Personal Loan:0'},
                                        {'x': [1,2,3],'y':[205,182,93] ,'type':'bar','name':'Personal Loan:1'}
                                    ],
                                    'layout':dict(
                                        {'title':'Education - Personal Loan '},
                                        xaxis={'title': 'Education'}
                                    )
                                }
                            )
                        ], className = 'col-6'),

                        html.Div([
                            # py.iplot(fig, filename='grouped-bar')
                            # Change the bar mode
                        #     fig.update_layout(barmode='group')
                        # fig.show()
                            dcc.Graph(
                                id='cdaccount-chart',
                                figure={
                                    'data': [
                                        {'x': [0,1],'y':[4358,340],'type':'bar','name':'Personal Loan:0'},
                                        {'x': [0,1],'y':[162,140] ,'type':'bar','name':'Personal Loan:1'}
                                    ],
                                    'layout':dict(
                                        {'title':'CD Account - Personal Loan '},
                                        xaxis={'title': 'CD Account'}
                                    )
                                }
                            )
                        ], className = 'col-6')
                    ], className = 'row'),    

                    html.Div(children = [
                        html.Div([
                            dcc.Graph(
                                id='Mortgage-chart',
                                figure={
                                    'data': [{
                                    'x': df['Personal Loan'],
                                    'y': df['Mortgage'],
                                    'type': 'box',
                                    'name' :'Personal Loan'
                                    }],
                                    'layout':dict(
                                        {'title':'Mortgage - Personal Loan '},
                                        xaxis={'title': 'Personal Loan'}
                                    )
                                }
                            )
                        ],className = 'col-5')
                    ],className = 'row')
                ])
            ]),


            dcc.Tab(value='Prediction', label='ML Prediction', children =[
                html.Div(children = [
                    html.Div(children =[
                        html.P('Income'),
                        dcc.Input(
                        id='loan-income', 
                        type = 'number', 
                        value = ''
                    )], className = 'col-3'),
                    
                    html.Div(children =[
                        html.P('CCAvg'),
                        dcc.Input(
                        id='loan-ccavg', 
                        type = 'number', 
                        value = ''
                    )], className = 'col-3'),
                    
                    html.Div(children =[
                        html.P('Education'),
                        dcc.Input(
                        id='loan-education', 
                        type = 'number',
                        value = ''
                    )], className = 'col-3'),

                    html.Div(children =[
                        html.P('Mortgage'),
                        dcc.Input(
                        id='loan-mortgage', 
                        type = 'number', 
                        value = ''
                    )], className = 'col-3'),

                    html.Div(children =[
                        html.P('CD Account'),
                        dcc.Input(
                        id='loan-cd_account', 
                        type = 'number', 
                        value = ''
                    )], className = 'col-3'),

              ], className = 'row'),
                html.Br(),
                html.Button('Predict', id = 'loan-button', className = 'col-1'),
                html.Div(id = 'Hasil-predict')
            ]),
        ])
        ])])
])

@app.callback(
    Output(component_id = 'div-table', component_property = 'children'),
    [Input(component_id = 'filter', component_property = 'n_clicks')],
    [State(component_id = 'filter-personal_loan', component_property = 'value'),
    State(component_id = 'filter-row', component_property = 'value')]
)

def update_table(n_clicks, personal_loan, row):
    if personal_loan == '':
        children = [generate_table(df, page_size = row)]
    else:
        children = [generate_table(df[df['Personal Loan'] == personal_loan ], page_size = row)]            
    return children

@app.callback(
    Output(component_id ='Hasil-predict', component_property = 'children'),
    [Input(component_id ='loan-button', component_property = 'n_clicks')],
    [State(component_id ='loan-income', component_property = 'value'),
    State(component_id ='loan-ccavg', component_property = 'value'),
    State(component_id ='loan-education', component_property = 'value'),
    State(component_id ='loan-mortgage', component_property = 'value'),
    State(component_id ='loan-cd_account', component_property = 'value')]
)
def predict_model(n_clicks, income, ccavg, education, mortgage,cd_account):
    if income == '' or ccavg =='' or education == '' or mortgage == '' or cd_account == '':
        return html.Center(html.H1('Please Fill all the value'))
    else:
        prediction = log_model.predict(np.array([income, ccavg, education, mortgage,cd_account]).reshape(1,-1))
        prob =log_model.predict_proba(np.array([income, ccavg, education, mortgage,cd_account]).reshape(1,-1))[0][prediction]
        if prediction == 0:
            return html.Center(html.H1('Customer will reject personal loan with probability {}'.format(round(prob[0], 2))))
        else:
            return html.Center(html.H1('Customer will accept personal loan with probability {}'.format(round(prob[0], 2))))


if __name__ == '__main__':
    app.run_server(debug=True)
