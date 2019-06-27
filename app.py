import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import os
import cv2
from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from sklearn.cluster import KMeans
import sys
import argparse
from ip import img_label
from base64 import b64encode
from flask import Flask, Response


#plt.style.use('seaborn-white')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

app.scripts.config.serve_locally = True

furnap = './static/furnap.jpg'
'''
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imread('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video_feed')



def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
'''
app.layout = html.Div([
    html.Img(src='data:image/jpg;base64,{}'.format(b64encode(open('{}'.format(furnap), 'rb').read()).decode())),
    html.Label('Living Room Furniture'),
                       dcc.Dropdown(
                                id='fur',
                                options=[{'label': 'Coffee Table', 'value': 'coffee_table'},
                                {'label': 'Sofa, Couch & Love Seats ', 'value': 'sofa'},
                                {'label': 'Chairs', 'value': 'chairs'}],
                                    value='coffee_table'
                                    ),

                       dcc.Upload(
                                  id='upload-image',
                                  children=html.Div([
                                                     'Drag and Drop or ',
                                                     html.A('Select Files')
                                                     ]),
                                  style={
                                  'width': '100%',
                                  'height': '60px',
                                  'lineHeight': '60px',
                                  'borderWidth': '1px',
                                  'borderStyle': 'dashed',
                                  'borderRadius': '5px',
                                  'textAlign': 'center',
                                  'margin': '10px'
                                  },
                                  # Allow multiple files to be uploaded
                                  multiple=True
                                  ),
                       html.Div(id='output-image-upload'),
                       html.Div(id='output-matching-image-upload'),
                       #html.Button(id='camera'),
                       #html.Img(src="/video_feed")
                       ])

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
                     #html.Div('Raw Content'),
                     # html.Pre(contents[0:200] + '...', style={
                     # 'whiteSpace': 'pre-wrap',
                     #  'wordBreak': 'break-all'
                     #      })
        ])


@app.callback(Output('output-image-upload','children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
                    parse_contents(c, n, d) for c, n, d in
                    zip(list_of_contents, list_of_names, list_of_dates)]
        return children

#app.callback(Output('intermediate-value', 'children'), [Input('upload-image', 'children')])
def links_func(link):
    return html.Div([
                     html.H5("Matching images"),
                     #html.H5('{}'.format(name)),
                     # HTML images accept base64 encoded strings in the same format
                     # that is supplied by the upload
                     #dcc.Link('Links to Matching Furniture', href='{}'.format(link), target='_blank'),
                     html.A('Link', href='{}'.format(link), target='_blank'),
                     #html.Img(src='data:image/jpeg;base64,{}'.format(name)),
                     html.Hr()
                ])
def parse_names(name):
    return html.Div([
                     html.H5("Matching images"),
                     #html.H5('{}'.format(name)),
                     # HTML images accept base64 encoded strings in the same format
                     # that is supplied by the upload
                     html.Img(src='data:image/jpeg;base64,{}'.format(name)),
                     html.Hr()
                     ])
@app.callback(Output('output-matching-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])
def img_label_display(list_of_contents, list_of_names):
    l_image = []
    if list_of_names is not None:
        l_image = img_label(list_of_names[0])[0]
        l_link = img_label(list_of_names[0])[1]
        encoded_image = [b64encode(open(l_image[i], 'rb').read()).decode() for i in range(len(l_image))]
        #print(l_image)
        #encoded_image = [b64encode(open(l_link[i], 'rb').read()).decode() for i in range(len(l_link))]
        #print(encoded_image)
        children = [html.Div([links_func(u) for u in l_link]),html.Div([parse_names(encoded_image[i]) for i in range(len(encoded_image))])]
        print(l_image)
        return children
                       





'''
def img_label_display(list_of_contents, list_of_names):
    encoded_image =[]
    if list_of_names is not None:
    print(img_label(list_of_names[0]))
    for img in img_label(list_of_names[0]):
        encoded_image = encoded_image.append([b64encode(open(img, 'rb').read()).decode()])
        print(list_of_names)
        print(encoded_image)
        children = [parse_names(n) for n in encoded_image]
        print(img_label(list_of_names[0]))
        return children
        
def update_histogram(figure):
    # Retrieve the image stored inside the figure
    enc_str = figure['layout']['images'][0]['source'].split(';base64,')[-1]
    # Creates the PIL Image object from the b64 png encoding
    im_pil = drc.b64_to_pil(string=enc_str)
    
    return show_histogram(im_pil)
def show_histogram(image):
    def hg_trace(name, color, hg):
        line = go.Scatter(
            x=list(range(0, 256)),
            y=hg,
            name=name,
            line=dict(color=(color)),
            mode='lines',
            showlegend=False
        )
        fill = go.Scatter(
            x=list(range(0, 256)),
            y=hg,
            mode='fill',
            name=name,
            line=dict(color=(color)),
            fill='tozeroy',
            hoverinfo='none'
        )
                          
        return line, fill
    hg = image.histogram()

    if image.mode == 'RGBA':
        rhg = hg[0:256]
        ghg = hg[256:512]
        bhg = hg[512:768]
        ahg = hg[768:]
        
        data = [
                *hg_trace('Red', '#FF4136', rhg),
                *hg_trace('Green', '#2ECC40', ghg),
                *hg_trace('Blue', '#0074D9', bhg),
                *hg_trace('Alpha', 'gray', ahg)
                ]
            
        title = 'RGBA Histogram'
                
    elif image.mode == 'RGB':
        # Returns a 768 member array with counts of R, G, B values
        rhg = hg[0:256]
        ghg = hg[256:512]
        bhg = hg[512:768]
            
        data = [
                    *hg_trace('Red', '#FF4136', rhg),
                    *hg_trace('Green', '#2ECC40', ghg),
                    *hg_trace('Blue', '#0074D9', bhg),
        ]

        title = 'RGB Histogram'

    else:
        data = [*hg_trace('Gray', 'gray', hg)]
        
        title = 'Grayscale Histogram'

    layout = go.Layout(
                           title=title,
                           margin=go.Margin(l=35, r=35),
                           legend=dict(x=0, y=1.15, orientation="h")
                           )

    return go.Figure(data=data, layout=layout)
'''
   
if __name__ == '__main__':
    app.run_server(debug=True)
