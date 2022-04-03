# Leaf Reconstruction
Harry Freeman (hfreeman), Gerard Maggiolino (gmaggiol), and David Russell (davidrus)
For a CMU course project in 16-889: Learning for 3D Vision, Spring 2022.

# Setup
Dependencies are managed by [poetry](https://python-poetry.org/). This provides pinned dependencies, powerful solving, and easy `pip` deployment. To install, follow the directions [here](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions). Once you've completed this, you can run `poetry install` to setup all the required dependencies. This command can be repeated any time dependencies change. `poetry` manages a virtual environment with all the dependencies. To activate it, run `poetry shell` within this project directory.

This project uses [dvc](https://dvc.org/) to manage the data, which should be installed by poetry as a development dependency. Once it is installed, you can run `dvc pull` to acquire the used by this project. The first time you do this, it will ask you to authenticate that you have access to the Google Drive where the data is stored. Follow the steps to allow all requested permissions. 

# DVC
Overview of how to add data to `dvc`.
```
dvc add <files>
# run the git command that shows up and commit
git push # This may ask you to set up permission with Google drive. You need the GDrive data folder shared with you.
dvc push
```
 