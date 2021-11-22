# LJPTagger
1. Train classifier (circa 50 epochs I believe): class_train.py
2. Pre-train adversarial (I do ~20-30 epochs, but normally it converges rather quickly): adv_train.py
3. combined training (not sure we can do 200, but let's try): combined_train.py
4a. Look at the metrics of the model saved in .txt, pick a model that you like
4b. Make scores using the selected model: make_scores.py
5. Make plots. I've created a new plotting macro that seems easier to use (but may be it's because I created it). But it's surely quicker
