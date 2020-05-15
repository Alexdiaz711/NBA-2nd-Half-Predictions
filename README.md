# <div align="center">NBA 2nd-Half Predictions</div>
#### <div align="center">by Alex Diaz-Clark</div>
Developing a supervised machine learning algorithm to predict which team will score more points in the second-half of NBA games, including a cost-benefit analysis, and betting simulation.

## Background

Sports betting was once a hobby for sports enthusiasts looking to make the games a little more exciting. But with the emergence of machine learning algorithms, predictive models are being built with the aim of predicting the results of sporting events in order to generate a small expected-profit for every bet. There are professional sports bettors all over the world that try to exploit small statistical advantages over a large volume of bets in order to turn a profit.

The goals of this project are as follows:
* Develop a machine learning model that was be used to predict which team will score more points in the second-half of National Basketball Association (NBA) games. 
* Use the model to predict the second-half result of games during the 2015, 2016, and 2017 NBA seasons. 
* Use those predictions, along with the historical betting odds for the second-half moneyline bet, in a cost/benefit analysis to develop a betting strategy for using the model. 
* Test the betting strategy on the 2018 and 2019 NBA seasons in a betting simulation.

## The Data

Data for this project was collected from multiple sources:
* In-game statistics from the first-half of NBA games from 

## Tuning Models

## Model Selection

## Cost/Benefit Analysis

## Betting Simulation

The bet that this model was built to exploit for a profit is the second-half moneyline bet. With this bet, a bettor can place a wager on either team to score more points than their opponent in the second-half, including overtime, with the actual winner of the game having no impact on the result of the bet. It is pretty stright-forward, if you pick team A, and team A scores more points after halftime, you win the bet. If team B scores more points after halftime, you lose the bet. If the teams score the same amount of points after halftime, the full bet is returned to the bettor.

While deciding who wins the bet is pretty straight forward, the betting odds are not. The odds are listed in the following format:

* "-150" means that to win $100 profit, the bettor must wager $150.
* "+150" means that if the bettor wagers $100 to win $150 profit.

Odds that are negative, are typically reserved for the team that the sportbook favors to score more points after halftime. Odds that are positive, indicate that the sportsbook believes that team will score less points than their opponent after halftime. This can be confusing, so for the sake of the reader, and to make calculations more stright-forward in the simulation, the odds have all been converted to a format which is strictly the potential profit from a $100 bet. For example:

* "-150" is converted to "$66.67"
* "+150" is converted to "$150.00"

## Conclusions

## Disclaimer
Sports betting, or gambling of any sort, should not be taken lightly. The models and strategies recommended here are simply for educational purposes. USE AT YOUR OWN RISK. If you, or someone you know, might have a gambling problem, please call the National Problem Gambling Helpline at 1-800-522-4700
