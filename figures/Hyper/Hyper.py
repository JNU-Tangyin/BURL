import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['font.size'] = 15

task_files = {
    'S-URL':'../../Experiment/Hyper_Parameters/Spam/Spam_evaluation_results_xgb.csv',
    'NPJ':'../../Experiment/Hyper_Parameters/News/News_evaluation_results_xgb.csv',
    'M-URL':'../../Experiment/Hyper_Parameters/Malicious/Malicious_evaluation_results_xgb.csv',
    'URL-C':'../../Experiment/Hyper_Parameters/Classification/Classification_evaluation_results_xgb.csv',
    'URL-A':'../../Experiment/Hyper_Parameters/App/App_evaluation_results_xgb.csv',
}

sns.set_style("whitegrid")

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

# Num Epochs
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, metric in enumerate(metrics):
    for task_name, file_path in task_files.items():
        data = pd.read_csv(file_path)
        sns.lineplot(ax=axes[i], data=data, x='Num Epochs', y=metric, marker='o', label=task_name)
    axes[i].set_title(metric.capitalize(), fontsize=20)
    axes[i].set_xlabel('Num Epochs')
    if i == 0:
        axes[i].set_ylabel('Value')
    axes[i].legend(title='Task', fontsize=12, loc = 'lower right')
plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.tight_layout()
plt.savefig('Num_Epochs.pdf')
plt.show()

# Output Dim
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, metric in enumerate(metrics):
    for task_name, file_path in task_files.items():
        data = pd.read_csv(file_path)
        sns.lineplot(ax=axes[i], data=data, x='Output Dim', y=metric, marker='o', label=task_name)
    axes[i].set_title(metric.capitalize(), fontsize=20)
    axes[i].set_xlabel('Output Dim')
    if i == 0:
        axes[i].set_ylabel('Value')
    axes[i].legend(title='Task', fontsize=12, loc = 'lower right')
plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.tight_layout()
plt.savefig('Output_Dim.pdf')
plt.show()

# Margin
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, metric in enumerate(metrics):
    for task_name, file_path in task_files.items():
        data = pd.read_csv(file_path)
        sns.lineplot(ax=axes[i], data=data, x='Margin', y=metric, marker='o', label=task_name)
    axes[i].set_title(metric.capitalize(), fontsize=20)
    axes[i].set_xlabel('Margin')
    if i == 0:
        axes[i].set_ylabel('Value')
    axes[i].legend(title='Task', fontsize=12, loc = 'lower right')
plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.tight_layout()
plt.savefig('Margin.pdf')
plt.show()

# Batch Size
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, metric in enumerate(metrics):
    for task_name, file_path in task_files.items():
        data = pd.read_csv(file_path)
        sns.lineplot(ax=axes[i], data=data, x='Batch Size', y=metric, marker='o', label=task_name)
    axes[i].set_title(metric.capitalize(), fontsize=20)
    axes[i].set_xlabel('Batch Size')
    if i == 0:
        axes[i].set_ylabel('Value')
    axes[i].legend(title='Task', fontsize=12, loc = 'lower left')
plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.tight_layout()
plt.savefig('Batch_Size.pdf')
plt.show()