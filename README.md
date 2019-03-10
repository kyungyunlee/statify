# statify
Music taste analysis with Spotify Web API 

link to website -> [http://statify-with-spotify.herokuapp.com](http://statify-with-spotify.herokuapp.com) 


### Functions that use Spotify Web API for analysis + plotting is in `login/views.py`

#### Requirements
* see `requirement.txt`
* authorization to access/modify user data (go check the Spotify API website) 
  * couldn't do this in python; relevant code can be found in `login/templates/index.html`, `login/templates/login_callback.html`. 

#### Slides from PyCascades2019 talk
* [So tell me, what is your music taste?](https://kyungyunlee.github.io/assets/post_images/20190224/pycascade_upload.pdf)


p.s. Code is not optimized, so the data retrieval and plotting make the app slow. (Especially the recomendation function is very slow). 
Please just look at it as a reference for using the web API to get music data for your own projects :) 
Let me know if you make anything fun! 
