# reference https://liangfgithub.github.io/Proj/MovieRecommendApp.html
from logging import debug
from typing import Dict

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, dcc, html
from dash.dependencies import ALL, State

from myfuns import (genres, get_displayed_movies, get_top10_popular_movies, get_recommended_movies)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], 
               suppress_callback_exceptions=True)
server = app.server

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H3("Movie Recommender", className="display-8"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("System I - Top10 Movies by Ave Rating", href="/", active="exact"),
                #dbc.NavLink("System II - Item Based Collaborative Filtering (with or without user rating input)", href="/system-2", active="exact"),
                dbc.NavLink("System II - Customized Recommendation by User Input", href="/system-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])

def render_page_content(pathname):
    if pathname == "/":
        movies = get_top10_popular_movies()
        return html.Div(
            [
                html.H1("Top 10 Popular Movies by Average Rating"),
                html.Div(
                    [

                                get_movie_card(movie, with_rating=False)

                                for idx, movie in movies.iterrows()
                    ],
                            className="row row-cols-1 row-cols-5",
                            id="rating-movies",
                ),
            ]
        )
    elif pathname == "/system-2":
        movies  = get_displayed_movies()
        return html.Div(
            [
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H1("Rate zero or some movies below to"),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        children=[
                                            "Get recommendations ",
                                            html.I(className="bi bi-emoji-heart-eyes-fill"),
                                        ],
                                        size="lg",
                                        className="btn-success",
                                        id="button-recommend",
                                    ),
                                    className="p-0",
                                ),
                            ],
                            className="sticky-top bg-white py-2",
                        ),
                        html.Div(
                            [
                                get_movie_card(movie, with_rating=True)
                                for idx, movie in movies.iterrows()
                            ],
                            className="row row-cols-1 row-cols-5",
                            id="rating-movies",
                        ),
                    ],
                    id="rate-movie-container",
                ),
                html.H1(
                    "Your recommendations", id="your-recommendation",  style={"display": "none"}
                ),
                dcc.Loading(
                    [
                        dcc.Link(
                            "Try again", href="/system-2", refresh=True, className="mb-2 d-block"
                        ),
                        html.Div(
                            className="row row-cols-1 row-cols-5",
                            id="recommended-movies",
                        ),
                    ],
                    type="circle",
                ),
            ]
        )


    
def get_movie_card(movie, with_rating=False):
    
    return html.Div(
        dbc.Card(
            [
                dbc.CardImg(
                    src=f"https://liangfgithub.github.io/MovieImages/{movie.MovieID}.jpg?raw=true",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        html.H6(movie.title, className="card-title text-center"),
                    ]
                ),
            ]
            + (
                [
                    dcc.RadioItems(
                        options=[
                            {"label": "1", "value": "1"},
                            {"label": "2", "value": "2"},
                            {"label": "3", "value": "3"},
                            {"label": "4", "value": "4"},
                            {"label": "5", "value": "5"},
                        ],
                        className="text-center",
                        id={"type": "movie_rating", "MovieID": movie.MovieID},
                        inputClassName="m-1",
                        labelClassName="px-1",
                    )
                ]
                if with_rating
                else []
            ),
            className="h-100",
        ),
        className="col mb-4",
    )
    
@app.callback(
    Output("rate-movie-container", "style"),
    Output("your-recommendation", "style"),
    [Input("button-recommend", "n_clicks")],
    prevent_initial_call=True,
)    
def on_recommend_button_clicked(n):
    return {"display": "none"}, {"display": "block"}

@app.callback(
    Output("recommended-movies", "children"),
    [Input("rate-movie-container", "style")],
    [
        State({"type": "movie_rating", "MovieID": ALL}, "value"),
        State({"type": "movie_rating", "MovieID": ALL}, "id"),
    ],
    prevent_initial_call=True,
)

def on_getting_recommendations(style, ratings, ids):
    rating_input = {
            f"m{ids[i]['MovieID']}": int(rating) for i, rating in enumerate(ratings) if rating is not None
        }
    
    #print('rating input', rating_input)
    recommended_movies = get_recommended_movies(rating_input)
    
    #print(recommended_movies)
    return [get_movie_card(movie) for idx, movie in recommended_movies.iterrows()]



@app.callback(
    Output("button-recommend", "disabled"),
    Input({"type": "movie_rating", "MovieID": ALL}, "value"),
)
def update_button_recommened_visibility(values):
    #return not list(filter(None, values))
    return False

if __name__ == "__main__":
    app.run_server(port=8080, debug=True)
