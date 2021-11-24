# LJPTagger
1. Train classifier (circa 50 epochs I believe): class_train.py
2. Pre-train adversarial (I do ~20-30 epochs, but normally it converges rather quickly): adv_train.py
3. combined training (not sure we can do 200, but let's try): combined_train.py
4. Look at the metrics of the model saved in .txt, pick a model that you like
5. Make scores using the selected model: make_scores.py
6. Make plots. I've created a new plotting macro that seems easier to use (but may be it's because I created it). But it's surely quicker



In order to run new code that reads config files, you need to:
1. Obviously, make sure that the paths and the options defined in the configuration files are the options that you would like to run

2. You need to do the following:

cd src/
python class_train.py configs/config_class_train.yaml  
python adv_train.py configs/config_adv_train.yaml  
python combined_train.py configs/config_combined_train.yaml  
python make_scores.py configs/config_make_scores.yaml  
