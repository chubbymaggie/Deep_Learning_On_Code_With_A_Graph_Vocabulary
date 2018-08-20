from experiments.evaluate_models_for_experiment import evaluate_models_for_experiment
from experiments.run_command_on_remote import run_command_on_remote

experiment_run_log_id = ''
skip_s3_sync = False
test = False

if __name__ == '__main__':
    list_of_kwargs = dict(list_of_kwargs=[
        dict(seed=5146,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='FITB_vocab_comparison',
             experiment_run_log_id=experiment_run_log_id,
             model_name='FITBFixedVocabGGNN',
             model_label='all_edge',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=('evaluate_FITB_accuracy',),
             model_params_to_load='best.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
        dict(seed=5146,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='FITB_vocab_comparison',
             experiment_run_log_id=experiment_run_log_id,
             model_name='FITBCharCNNGGNN',
             model_label='all_edge',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=('evaluate_FITB_accuracy',),
             model_params_to_load='best.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
        dict(seed=5146,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='FITB_vocab_comparison',
             experiment_run_log_id=experiment_run_log_id,
             model_name='FITBNameGraphVocabGGNN',
             model_label='all_edge',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=('evaluate_FITB_accuracy',),
             model_params_to_load='best.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
        dict(seed=5146,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='FITB_vocab_comparison',
             experiment_run_log_id=experiment_run_log_id,
             model_name='FITBFixedVocabGGNN',
             model_label='syntax_edge',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=('evaluate_FITB_accuracy',),
             model_params_to_load='best.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
        dict(seed=5146,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='FITB_vocab_comparison',
             experiment_run_log_id=experiment_run_log_id,
             model_name='FITBCharCNNGGNN',
             model_label='syntax_edge',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=('evaluate_FITB_accuracy',),
             model_params_to_load='best.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
        dict(seed=5146,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='FITB_vocab_comparison',
             experiment_run_log_id=experiment_run_log_id,
             model_name='FITBNameGraphVocabGGNN',
             model_label='syntax_edge',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=('evaluate_FITB_accuracy',),
             model_params_to_load='best.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
    ]
    )
    run_command_on_remote('local',
                          evaluate_models_for_experiment,
                          list_of_kwargs)
