# fed-cycle-flattener-research
Signal-strength analysis for yield curve flattener strategy

Flattener strategy uses a spread between 2y UST and the Fed Funds rate in order to determine if rate hikes are priced in. Benchmark is a buy and hold strategy of the 10Y US Treasury Bond every day at par. 


<img width="1188" height="640" alt="image" src="https://github.com/user-attachments/assets/3edf84e0-e76a-4d17-a5a5-363fbc4db14c" />


<img width="1189" height="540" alt="image" src="https://github.com/user-attachments/assets/07ba7c7c-041d-4de6-9589-dd7b256d3a82" />


[fed_cycle_vs_10y_buy_hold_analysis.pdf](https://github.com/user-attachments/files/33289601/fed_cycle_vs_10y_buy_hold_analysis.pdf)

Strategy gains returns and sharpe when a 25bp threshold is applied to be able to discern when to trade or when to stay in cash. Better rate of return per unit of risk then taking every spread greater than 0, however, the sharpe of the strategy is still relatively low. 

<img width="2016" height="1180" alt="image" src="https://github.com/user-attachments/assets/2028af08-e743-4f03-9b7e-1ab7beae8e2a" />

However, performance does not rise with the increase of the threshold, it behaves more logarithmically, so as the threshold increases, there is diminishing marginal returns. However, larger thresholds do relatively decrease your risk, but also decreases your returns. 

<img width="2160" height="777" alt="image" src="https://github.com/user-attachments/assets/bc59042c-e340-465f-bef5-feaf94da7aa0" />

<img width="1584" height="1152" alt="image" src="https://github.com/user-attachments/assets/75e8da63-5e74-4fcb-940d-a9afef49afec" />
