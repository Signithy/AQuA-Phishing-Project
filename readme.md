Setup

python -m venv .venv
..venv\Scripts\activate
pip install --upgrade pip
pip install -r aqua/requirements.txt
pip install pandas scikit-learn matplotlib

Cleanlab Label Cleaning

ENRON:
python projectcode\code\label_quality_cl.py --input_csv projectcode\data\enron_scored.csv --output_scored_csv projectcode\data\enron_cl_scored.csv --output_clean_csv projectcode\data\enron_cl_labelclean.csv --drop_fraction 0.05

NASER:
python projectcode\code\label_quality_cl.py --input_csv projectcode\data\naser_scored.csv --output_scored_csv projectcode\data\naser_cl_scored.csv --output_clean_csv projectcode\data\naser_cl_labelclean.csv --drop_fraction 0.05

TWENTE:
python projectcode\code\label_quality_cl.py --input_csv projectcode\data\twente_scored.csv --output_scored_csv projectcode\data\twente_cl_scored.csv --output_clean_csv projectcode\data\twente_cl_labelclean.csv --drop_fraction 0.05

Logistic Regression Experiments

Single LR run:
python projectcode\code\train_logreg.py --train_csv projectcode\data\enron_cl_labelclean.csv --test_csv projectcode\data\enron_raw.csv

Run all LR experiments:
python projectcode\code\logreg_all_results.py

BERT Experiments

Single BERT run:
python projectcode\code\train_bert.py --train_csv projectcode\data\enron_cl_labelclean.csv --test_csv projectcode\data\enron_raw.csv --epochs 1 --batch_size 8 --max_length 64

Run all BERT experiments:
python projectcode\code\bert_all_results.py --epochs 1 --batch_size 8 --max_length 64

Aggregate LR + BERT Results

python projectcode\code\aggregate_results.py

Outputs:
projectcode\data\combined_results_table.csv

Generate All Figures

python projectcode\code\plot_results.py

Outputs saved in:
projectcode\figures\