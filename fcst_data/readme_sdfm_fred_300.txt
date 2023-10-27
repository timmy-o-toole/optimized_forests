The following file contains forecasts from Daniele, Kronenburg & reinicke (2023) for 5 variables of the fred-md data.
Dates correspond rows "python_pipeline\data_out". Note the row denotes the most current date (12.2022) and the first row the ealierst. Estimation of all models starts 1975.
 
The following models are included:
[1] "adalasso"    "adalassoaic" "adalassobic" "arp"         "FM.bn1"      "FM.ed"       "FM.ts.bn1"  
[8] "FM.em.bn1"   "FM.ts.ed"    "FM.em.ed"    "FM.ts.saf"   "FM.em.saf"   "FM.saf1"     "FM.saf2"    
[15] "FM.saf3"     "FM.saf4"     "en"          "enaic"       "enbic"       "lasso"       "lassoaic"   
[22] "lassobic"    "rf"          "ridge"       "rw" 

I recommend to use the following (as many are to similiar):
 "arp", "FM.bn1", "FM.ts.ed" , "en", "rf", "ridge", "rw".

Column description:
values = Y_true - Y_pred