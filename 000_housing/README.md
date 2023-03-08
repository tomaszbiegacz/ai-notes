# House prices regression

Regression playbooks for "California Housing Prices" <https://github.com/ageron/handson-ml3>.

## Performance

PC

pc | processor | clock | threads/cores | TDP | RAM size | RAM speed | Notes
---|---|---|---|---|---|---|---
viao | Intel Core i7-4500U | 1.8 - 3 | 4/2 | 15 | 8 | DDR3/1600 | 100% pamięci, procesor na 100%
msi | Intel Core i7-4710HQ | 2.5 - 3.5 | 8/4 | 47 | 16 | DDR3/1600 | 70% pamięci, procesor na 100%
hp | AMD Ryzen 7 PRO 5850U | 1.9 - 4.4 | 16/8 | 15 | 32 | DDR4/3200 | 50% pamięci, procesor < 70%

[Azure](https://azure.microsoft.com/en-us/pricing/details/machine-learning/) - East US

series | vCPU | RAM | Price $/h | Notes
---|---|---|---|---
F4s_v2 | 4 | 8 | 0.169 | CPU 100%, RAM < 10%
F8s_v2 | 8 | 16 | 0.338 | CPU 100%, RAM < 5%

### Comparison

pc | RandomForestRegressor | GridSearchCV | RandomizedSearchCV
---|---|---|---|---
viao | 6:13 | 
msi | 3:29 | 5:41 | 4:30 
hp | 2:33 | 2:44 | 2:24
F4s_v2 | 2:55 | 4:16 | 1:21
F8s_v2 | 2:48 | 4:17 | 3:24
