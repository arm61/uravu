# Contributing

Thank you for your interest in contributing to the development of `uravu`. 
This, like many open source projects, needs the contribution of people "free-time" to thrive, so we are really happy that you have taken the time to help out!

To make contributions easy to manage, we would appreciate if you would follow a relatively standard contributing workflow. 

- Firstly, my a fork of the Github repository on your own Github account (this can be achieved by clicking the Fork buttomn on the top right of the project page)
- Make the changes that you would like to contribute, including tests
- Use [`black`](https://black.readthedocs.io) to format the code (note the line length of 79 characters)
```
black -l 79 uravu/*
```
- Check there are no style suggestions from [`flake8`](https://flake8.pycqa.org/)
```
flake8 uravu/*
```
- Run [`pytest`](https://docs.pytest.org/) to make sure that everything is passing
- Add your changes to your fork on a new branch (with a descriptive name)
```
git checkout -b my_brach
git add uravu/changed_file_1 uravu/changed_file_2 
git commit -m 'A sensible and descriptive message'
git push
```
- Having pushed your changes to your own fork, go to the the [`uravu`](https://github.com/arm61/uravu) project page and open the pull request.

We will try and provide detailed feedback on your pull request as quickly as possible.

Thanks for your interest!