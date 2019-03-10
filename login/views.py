from django.shortcuts import render, render_to_response, redirect
from django.template.loader import render_to_string
from django.template import RequestContext
from django.http import HttpResponse
from .forms import ApproveEventForm
from django.conf import settings

import spotipy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pygal
from pygal.style import Style
import mpld3
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import operator
from decouple import config


def make_feat_vec(track_feat):
    ''' makes audio feature vector with 8 audio features
    Args :
        track_feat : track data dictionary returned from spotipy.audio_features()
    Returns :
        feat: 8-dimensional vector (numpy array)
    '''

    cols = [
        'danceability',
        'energy',
        'loudness',
        'acousticness',
        'liveness',
        'valence',
        'instrumentalness',
        'tempo']
    feat = []
    for col in cols:
        if col == 'tempo':
            computed_feat = track_feat[col] / np.max(track_feat[col])
        elif col == 'loudness':
            computed_feat = (track_feat[col] + 60.0) / 60.0
        else:
            computed_feat = track_feat[col]

        feat.append(computed_feat)
    return feat


def make_dataframe(sp, tracks, playlist_id):
    '''
    Save audio features of playlist tracks into pandas dataframe.
    Data from Spotify API.
    Args:
        sp : spotipy's spotify web api object
        tracks : list of track dictionaries returned from Spotify web API track retrieval
        playlist_id : id of the currently analyzing playlist
    Returns :
        df : pandas dataframe
    '''

    # these are the audio features used for analysis
    cols = [
        'danceability',
        'energy',
        'loudness',
        'acousticness',
        'liveness',
        'valence',
        'instrumentalness',
        'tempo',
        'popularity']

    # get all the track ids
    track_ids = []
    for i in range(len(tracks)):
        track_ids.append(tracks[i]['track']['id'])

    # spotipy function to get audio features from list of track ids
    audio_features = sp.audio_features(track_ids)

    # fill dataframe
    df = pd.DataFrame(columns=cols +
                      ['playlist_id', 'name', 'artist', 'track_id'])
    for i in range(len(tracks)):
        df.loc[track_ids[i]] = pd.Series({'danceability': audio_features[i]['danceability'],
                                          'energy': audio_features[i]['energy'],
                                          'loudness': audio_features[i]['loudness'],
                                          'acousticness': audio_features[i]['acousticness'],
                                          'liveness': audio_features[i]['liveness'],
                                          'valence': audio_features[i]['valence'],
                                          'instrumentalness': audio_features[i]['instrumentalness'],
                                          'tempo': audio_features[i]['tempo'],
                                          'popularity': tracks[i]['track']['popularity'],
                                          'playlist_id': playlist_id,
                                          'name': tracks[i]['track']['name'],
                                          'artist': tracks[i]['track']['artists'][0]['name'],
                                          'track_id': tracks[i]['track']['id'],
                                          })

    return df


def analyze_user_top_tracks(sp, user_id, cols):
    '''
    Get some info on current user's music taste by looking at top 50 most listened tracks
    Args :
        sp : spotipy's Spotify web api object
        user_id : spotify user id of the current user
        cols : list of audio feature names
    Return :
        genre_chart : pygal chart of user top genres
        top_50_df : dataframe of user top 50 tracks audio features + metadata
        feature_importance_level : dictionary of feature name to level of importance to the current user
        taste_vector : current user's taste vector (8-dim)
    '''

    #############  Get user's top genre by looking at the genre data of user's
    user_top_artists = sp.current_user_top_artists(
        time_range='medium_term', limit=50)['items']
    genres = {}
    for u in user_top_artists:
        for g in u['genres']:
            try:
                genres[g] += 1
            except BaseException:
                genres[g] = 1

    genres = sorted(genres.items(), key=operator.itemgetter(1), reverse=True)

    # pygal bar chart for genre
    style = Style(background='white', plot_background='white')
    genre_chart = pygal.HorizontalBar(style=style, legend_at_bottom=True)
    genre_chart.force_uri_protocol = 'http'
    for top_genre in genres[:10]:
        genre_chart.add(top_genre[0], top_genre[1])
    genre_chart = genre_chart.render(
        is_unicode=True, force_uri_protocol='https')
    # genre_chart.render_to_file(settings.MEDIA_URL + 'login/my_genre.svg')

    #######################  User's top 50 track analyis #####################
    # creating a playlist is necessary b/c can't play tracks if it is not a
    # playlist!

    # get user top tracks
    user_top_tracks = sp.current_user_top_tracks(
        time_range='medium_term', limit=50)['items']

    # create new playlist of top 50 tracks
    top_track_ids = []
    for i, item in enumerate(user_top_tracks):
        top_track_ids.append(item['id'])
    # check if it exists already
    playlist_check = [i['name'] for i in sp.current_user_playlists()['items']]
    if 'Statify: My Top Tracks' in playlist_check:
        print('already exists')
        for i in sp.current_user_playlists()['items']:
            if i['name'] == 'Statify: My Top Tracks':
                top_track_pid = i['id']
        top_50_playlist = sp.user_playlist(user_id, top_track_pid)
        result = sp.user_playlist_replace_tracks(
            user_id, top_50_playlist['id'], top_track_ids)
    else:
        top_50_playlist = sp.user_playlist_create(
            user_id, 'Statify: My Top Tracks')
        result = sp.user_playlist_add_tracks(
            user_id, top_50_playlist['id'], top_track_ids)

    # get data and plot for user top 50 tracks
    top_50_df = make_dataframe(
        sp,
        sp.user_playlist_tracks(
            user_id,
            top_50_playlist['id'])['items'],
        top_50_playlist['id'])
    # request.session['user_top_50_dataframe'] = df.to_json()

    box_plot = pygal.Box(box_mode='stdev', style=style)
    box_plot.add('danceability', top_50_df['danceability'])
    box_plot.add('energy', top_50_df['energy'])
    box_plot.add('loudness', (top_50_df['loudness'] + 60) / 60)
    box_plot.add('acousticness', top_50_df['acousticness'])
    box_plot.add('liveness', top_50_df['liveness'])
    box_plot.add('valence', top_50_df['valence'])
    box_plot.add('instrumentalness', top_50_df['instrumentalness'])
    box_plot.add('popularity', top_50_df['popularity'] / 100)
    box_plot.add('tempo', top_50_df['tempo'] / np.max(top_50_df['tempo']))
    box_plot = box_plot.render(is_unicode=True, force_uri_protocol='https')
    # box_plot.render_to_file(settings.MEDIA_URL + 'login/my_boxplot.svg')

    # sort top 50 by features
    '''
    sort_by_danceability = top_50_df.sort_values('danceability', ascending=False) # danceable to not danceable
    # print (sort_by_danceability)
    sort_by_popularity = top_50_df.sort_values('popularity', ascending=False) # popular to not popular
    sort_by_tempo = top_50_df.sort_values('tempo', ascending=False) # fast to slow
    sort_by_energy = top_50_df.sort_values('energy', ascending=False) # energetic to not energetic
    '''

    # calculate mean, variance of each audio features
    mean_per_feat = []
    std_per_feat = []
    mean_per_feat_unscaled = []
    for col in cols:
        value_list = top_50_df[col].astype(float)
        unscaled = value_list
        if col == 'popularity':
            value_list = value_list / 100
        elif col == 'tempo':
            value_list = value_list / np.max(value_list)
        elif col == 'loudness':
            value_list = (value_list + 60) / 60

        mean_per_feat.append(round(np.mean(value_list), 3))
        mean_per_feat_unscaled.append(round(np.mean(unscaled), 3))
        std_per_feat.append(np.std(value_list))

    # get least to highest variance of audio features
    sorted_least_std_idx = np.argsort(std_per_feat)
    sorted_least_std = np.array(cols)[sorted_least_std_idx]
    feature_importance_level = []

    for i in range(len(sorted_least_std_idx)):
        mean_val_threshold = mean_per_feat[sorted_least_std_idx[i]]
        if mean_val_threshold < 0.4:
            thresh = 'low'
        elif mean_val_threshold < 0.6:
            thresh = 'medium'
        else:
            thresh = 'high'

        feature_importance_level.append([sorted_least_std[i],
                                         format(mean_per_feat[sorted_least_std_idx[i]],
                                                '.3f'),
                                         format(std_per_feat[sorted_least_std_idx[i]],
                                                '.3f'),
                                         thresh])

    # user taste vector is an average of the feature vectors of user's top 50
    # tracks
    taste_vector = mean_per_feat[:-1]
    # request.session['taste_vector'] = taste_vector

    return genre_chart, box_plot, top_50_playlist, top_50_df, feature_importance_level, taste_vector


def compute_recommendation(request):
    '''
    Rank (== recommend) songs from Spotify's 'New Music Friday' playlist, given user's music taste.
    '''
    accessToken = request.session['accessToken']
    sp = spotipy.Spotify(auth=accessToken)
    user_id = request.session['user_id']

    # from new release albums (too slow)
    '''
    new_release_feats = {}
    new_release_albums = sp.new_releases()
    track_counter = 0
    while new_release_albums:
        albums = new_release_albums['albums']

        for i, item in enumerate(albums['items']):
            if track_counter > 100:
                break
            print(albums['offset'] + i,item['name'])
            album_tracks = sp.album_tracks(item['id'])['items']
            for album_track in album_tracks:
                album_track_feat = sp.audio_features(album_track['id'])
                track_vec = make_feat_vec(album_track_feat[0])
                new_release_feats[album_track['id']] = track_vec
                track_counter += 1

        if albums['next']:
            new_release_albums = sp.next(albums)
        else:
            new_release_albums = None
    # request.session['recommendations'] = recommendations
    # return redirect('list_playlists')

    track_ids = list(new_release_feats.keys())
    track_feats = np.array(list(new_release_feats.values()))
    '''

    # Model all the tracks with 8 audio features from the 'New Music Friday'
    # playlist
    new_music_friday_id = '37i9dQZF1DX4JAvHpjipBk'
    new_tracks = sp.user_playlist_tracks(user_id, new_music_friday_id)['items']
    # print (new_tracks)
    new_track_ids = []
    new_track_names = []
    new_track_feats = []
    for new_track in new_tracks:
        try:
            track_feat = sp.audio_features(new_track['track']['id'])
            track_vec = make_feat_vec(track_feat[0])
            new_track_ids.append(new_track['track']['id'])
            new_track_names.append(new_track['track']['name'])
            new_track_feats.append(track_vec)
        except BaseException:
            pass

    new_track_feats = np.array(new_track_feats)

    # compare with user's taste vector
    # request.session['taste_vector'] contains computed user vector
    user_taste_vec = np.expand_dims(
        np.array(request.session['taste_vector']), 0)
    cosine_sim = cosine_similarity(user_taste_vec, new_track_feats)
    sorted_arg = np.argsort(-cosine_sim)[0]
    # return the 10 most 'likeable' tracks by the current user
    rec_track_names = np.array(new_track_names)[sorted_arg].tolist()[:10]
    rec_track_ids = np.array(new_track_ids)[sorted_arg].tolist()[:10]

    # Create new playlist of top 50 tracks
    # If it exists already, just update. Else, create.
    playlist_check = [i['name'] for i in sp.current_user_playlists()['items']]
    if 'Statify: New release Recommendation' in playlist_check:
        print('new release already exists')
        for i in sp.current_user_playlists()['items']:
            if i['name'] == 'Statify: New release Recommendation':
                top_track_pid = i['id']
        top_track_playlist = sp.user_playlist(user_id, top_track_pid)
        result = sp.user_playlist_replace_tracks(
            user_id, top_track_playlist['id'], rec_track_ids)
    else:
        top_track_playlist = sp.user_playlist_create(
            user_id, 'Statify: New release Recommendation')
        result = sp.user_playlist_add_tracks(
            user_id, top_track_playlist['id'], rec_track_ids)

    request.session['recommendation_playlist_id'] = top_track_playlist['id']

    return redirect('list_playlists')


def list_playlists(request):
    '''
    (first page after spotify login)
    Analyze current user's top 50 most listened tracks.
    And get all the playlist from the current user + Spotify's 3 global popular playlists.
    '''

    print("list playlists")

    # variables for current user
    accessToken = request.session['accessToken']
    sp = spotipy.Spotify(auth=accessToken)
    me = sp.me()
    username = me['display_name']
    user_id = me['id']
    request.session['username'] = username
    request.session['user_id'] = user_id

    cols = [
        'danceability',
        'energy',
        'loudness',
        'acousticness',
        'liveness',
        'valence',
        'instrumentalness',
        'tempo',
        'popularity']
    request.session['cols'] = cols

    # User top 50 tracks analysis
    genre_chart, box_plot, top_50_playlist, top_50_df, feat_importance_level, taste_vector = analyze_user_top_tracks(
        sp, user_id, cols)
    request.session['user_top_50_dataframe'] = top_50_df.to_json()
    request.session['taste_vector'] = taste_vector

    # if the user clicked recommendation button (see function
    # 'compute_recommendation' above), display the result, if not, display
    # nothing
    try:
        rec_pid = request.session['recommendation_playlist_id']
        # recommendation computed
        recommendation_computed = 'true'
    except BaseException:
        rec_pid = None
        recommendation_computed = 'false'

    ###########################  Get Spotify's popular playlists  ############

    global_top50_id = '37i9dQZEVXbMDoHDwVN2tF'  # 1. global top 50 playlist
    today_top_hits_id = '37i9dQZF1DXcBWIGoYBM5M'  # 2. today's top hits playlist
    kpop_id = '37i9dQZF1DX9tPFwDMOaN1'  # 3. kpop playlist
    global_top50 = sp.user_playlist(user_id, global_top50_id)
    today_top_hits = sp.user_playlist(user_id, today_top_hits_id)
    kpop = sp.user_playlist(user_id, kpop_id)

    ###########################  Get User's playlists  #######################
    playlists = sp.current_user_playlists(limit=50)['items']
    playlist_id_to_name = {}
    for playlist in playlists:
        playlist_id_to_name[playlist['id']] = playlist['name']

    playlist_id_to_name[global_top50_id] = global_top50['name']
    playlist_id_to_name[today_top_hits_id] = today_top_hits['name']
    playlist_id_to_name[kpop_id] = kpop['name']

    request.session['playlist_id_to_name'] = playlist_id_to_name

    return render(request, 'login/stats.html', {
        'username': username,
        'playlists': playlists,
        'user_top_playlist': top_50_playlist,
        'recommendation_playlist_id': rec_pid,
        # list of user top 50 tracks analysis per feature
        'feat_dict': feat_importance_level,
        'computed': recommendation_computed,
        'box_plot': box_plot,
        'genre_chart': genre_chart,
        'taste_vector': taste_vector,
        'df': top_50_df.to_dict('index'),
        'global50': global_top50,
        'todaytop': today_top_hits,
        'kpop': kpop})


def multiple_analysis(request):
    '''
    (page when user selected several playlists to analyze)
    Do analysis and plot graphs on user selected playlists.
    '''

    cols = [
        'danceability',
        'energy',
        'loudness',
        'acousticness',
        'liveness',
        'valence',
        'instrumentalness',
        'tempo',
        'popularity']

    playlist_id_to_name = request.session['playlist_id_to_name']

    if request.method == 'POST':
        checked = request.POST.getlist('choices')  # list of playlist ids
        accessToken = request.session['accessToken']
        sp = spotipy.Spotify(auth=accessToken)

        dfs = {}
        playlist_tracks = []
        for playlist_id in checked:

            # save track features in dataframe
            tracks = sp.user_playlist_tracks(
                sp.current_user()['id'], playlist_id)['items']
            df = make_dataframe(sp, tracks, playlist_id)
            dfs[playlist_id] = df

            # calculate mean, variance for each feature and find the lowest
            # variance features
            mean_per_feat = []
            std_per_feat = []
            for col in cols:
                value_list = dfs[playlist_id][col].astype(float)
                if col == 'popularity':
                    value_list = value_list / 100
                elif col == 'tempo':
                    value_list = value_list / np.max(value_list)
                elif col == 'loudness':
                    value_list = (value_list + 60) / 60

                mean_per_feat.append(np.mean(value_list))
                std_per_feat.append(np.std(value_list))

            sorted_least_std = np.argsort(std_per_feat)
            # print (std_per_feat)
            top3_least_std = np.array(cols)[sorted_least_std]
            high_or_low = []
            for i in range(len(sorted_least_std)):
                if i < 3:
                    if mean_per_feat[sorted_least_std[i]] < 0.4:
                        high_or_low.append('low')
                    elif mean_per_feat[sorted_least_std[i]] < 0.6:
                        high_or_low.append('medium')
                    else:
                        high_or_low.append('high')

            # for displaying tracks in the playlist
            playlist_track = {}
            playlist_track['name'] = playlist_id_to_name[playlist_id]
            playlist_track['tracks'] = list(df.index)
            playlist_track['id'] = playlist_id

            playlist_track['main_features'] = top3_least_std[:3]
            playlist_track['main_features_val'] = high_or_low

            playlist_tracks.append(playlist_track)
            # analysis_result, box_plot = perform_analysis(df)

        # calculate mean and variance
        playlist_means = []
        playlist_stds = []
        for playlist_id in checked:

            mean_per_feat = []
            std_per_feat = []
            for col in cols:
                value_list = dfs[playlist_id][col].astype(float)
                if col == 'popularity':
                    value_list = value_list / 100
                elif col == 'tempo':
                    value_list = value_list / np.max(value_list)
                elif col == 'loudness':
                    value_list = (value_list + 60) / 60

                mean_per_feat.append(np.mean(value_list))
                std_per_feat.append(np.std(value_list))
            playlist_means.append(mean_per_feat)
            playlist_stds.append(std_per_feat)

            print("std", std_per_feat)
            print(np.argsort(std_per_feat))
            sorted_least_std = np.argsort(std_per_feat)
            print(
                playlist_id_to_name[playlist_id],
                "top 3 least var feature",
                np.array(cols)[sorted_least_std])

        playlist_means = np.array(playlist_means)
        playlist_stds = np.array(playlist_stds)
        # For each playlist, find the smallest variance
        # ex. playlist A is more danceable

        # draw histogram of each features for selected playlists
        result = {}
        for col in cols:
            result[col] = {}
            fig = plt.figure(figsize=(3, 3,))  # an empty figure with no axes
            hist_plot = None
            sns.set(color_codes=True)
            sns.set(style="white", palette="muted")
            # sns.distplot(df)
            for playlist_id in checked:
                if col not in ['popularity', 'tempo', 'loudness']:
                    hist_plot = sns.distplot(
                        dfs[playlist_id][col].astype(float),
                        bins=np.linspace(
                            0.0,
                            1.0,
                            num=10),
                        hist=True,
                        label=playlist_id_to_name[playlist_id],
                        kde_kws={
                            "shade": True})
                elif col == 'popularity':
                    hist_plot = sns.distplot(
                        dfs[playlist_id][col].astype(float),
                        bins=np.linspace(
                            0,
                            100,
                            num=15),
                        hist=True,
                        label=playlist_id_to_name[playlist_id],
                        kde_kws={
                            "shade": True})
                elif col == 'tempo':
                    hist_plot = sns.distplot(
                        dfs[playlist_id][col].astype(float),
                        bins=np.linspace(
                            0,
                            250,
                            num=15),
                        hist=True,
                        label=playlist_id_to_name[playlist_id],
                        kde_kws={
                            "shade": True})
                elif col == 'loudness':
                    hist_plot = sns.distplot(
                        dfs[playlist_id][col].astype(float),
                        bins=np.linspace(
                            -60.0,
                            0.0,
                            num=15),
                        hist=True,
                        label=playlist_id_to_name[playlist_id],
                        kde_kws={
                            "shade": True})

                # hist_plot = sns.distplot(dfs[playlist_id][col].astype(float))

            hist_plot.set_title(col, fontsize=20)
            fig.axes.append(hist_plot)
            hist_plot.figure = fig
            fig.add_axes(hist_plot)
            plt.legend()
            html_plot = mpld3.fig_to_html(fig)
            result[col]['plot'] = html_plot

    # draw boxplot of selected playlists
    boxplots = []
    style = Style(background='white', plot_background='white')
    boxplot_counter = 0
    for playlist_id in checked:
        box_plot = pygal.Box(box_mode='stdev', style=style)
        box_plot.force_uri_protocol = 'http'
        box_plot.title = playlist_id_to_name[playlist_id]
        box_plot.add('danceability', dfs[playlist_id]['danceability'])
        box_plot.add('energy', dfs[playlist_id]['energy'])
        box_plot.add('loudness', (dfs[playlist_id]['loudness'] + 60) / 60)
        box_plot.add('acousticness', dfs[playlist_id]['acousticness'])
        box_plot.add('liveness', dfs[playlist_id]['liveness'])
        box_plot.add('valence', dfs[playlist_id]['valence'])
        box_plot.add('instrumentalness', dfs[playlist_id]['instrumentalness'])
        box_plot.add('popularity', dfs[playlist_id]['popularity'] / 100)
        box_plot.add(
            'tempo',
            dfs[playlist_id]['tempo'] /
            np.max(
                dfs[playlist_id]['tempo']))
        box_plot = box_plot.render(is_unicode=True, force_uri_protocol='https')
        # box_plot.render_to_file(settings.MEDIA_URL  + 'login/'+ str(boxplot_counter) + '.svg')
        boxplots.append(box_plot)
        boxplot_counter += 1

    # pca on all the audio features in the selected playlists
    for i in range(len(checked)):
        if i == 0:
            total_df = dfs[checked[i]]
        else:
            total_df = pd.concat([total_df, dfs[checked[i]]], axis=0)

    # exclude 'popularity'
    cols_minus_pop = cols.copy()
    cols_minus_pop.remove('popularity')
    x = total_df.loc[:, cols_minus_pop].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(
        data=principal_components,
        index=total_df.index.tolist(),
        columns=[
            'pc1',
            'pc2'])
    final_df = pd.concat(
        [principal_df, total_df[['playlist_id', 'name', 'artist']]], axis=1)

    scatter_plot = pygal.XY(stroke=False, style=style)
    for playlist_id in checked:
        # print (final_df.loc[final_df['playlist_id'] == playlist_id].loc[:, 'pc1':'pc2'].values)
        val = final_df.loc[final_df['playlist_id'] == playlist_id]
        list_of_val = []
        for index, row in val.iterrows():
            curr_val = {}
            curr_val['value'] = [row['pc1'], row['pc2']]
            curr_val['label'] = row['name'] + ' - ' + row['artist']
            list_of_val.append(curr_val)
        scatter_plot.add(playlist_id_to_name[playlist_id], list_of_val)
    pcaplot = scatter_plot.render(is_unicode=True, force_uri_protocol='https')
    # scatter_plot.render_to_file(settings.MEDIA_URL + 'login/pca_plot.svg')

    # sort by features
    # what are the saddest songs?
    # what are the least popular songs?

    return render(request,
                  'login/playlist_analysis.html',
                  {'num_playlists': len(checked),
                   'playlist_tracks': playlist_tracks,
                   'result': result,
                   'boxplots': boxplots,
                   'pcaplot': pcaplot})


def index(request):
    client_id = config('SPOTIPY_CLIENT_ID')
    return render(request, 'login/index.html', {'cid': client_id})


def login_callback(request):
    ''' Callback after successful Spotify login '''
    print("hello")
    return render(request, 'login/login_callback.html', {})


def main(request):
    ''' function after spotify login to get the accessToken '''
    # remove session variables
    try:
        del request.session['recommendation_playlist_id']
    except BaseException:
        pass

    if request.is_ajax():
        accessToken = request.POST.get("spAccessToken")
        request.session['accessToken'] = accessToken
        return redirect(index)
    else:
        return HttpResponse('no token recieved :( ')
