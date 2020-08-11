# similar-player-finder
Finds similar players based on particular stats of given player

## Credits
- All credit goes to [Parth Athale](https://twitter.com/ParthAthale)
- This is copied from his [original repo](https://github.com/parth1902/PCA_Player_Finder)
**I re-wrote it more coherently/elegantly as it was quite spaghetti when I first encountered it. I claim no credit for this idea. I solely did this for better understanding of it's working.**

## Usage
- Install dependencies with `pip install -r requirements.txt`
- You can search for the exact player/team names as follows
```
> cd src
> python
>>> import utils
>>> utils.search_player(name="davies")
```
- Feed your inputs at `inputs/user_inputs.csv`. Use precise names for players/teams.
- Run the script using `cd src` followed by `python run.py`