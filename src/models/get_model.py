from src.models import baselinemodels

def get_model(model_kwargs):
    if model_kwargs['model'] == 'BaselineCNN':
        model = baselinemodels.BaselineCNN(**model_kwargs)
    elif model_kwargs['model'] == 'BaselineCNNV2':
        model = baselinemodels.BaselineCNNV2(**model_kwargs)
    elif model_kwargs['model'] == 'AttBiLSTM':
        model = baselinemodels.AttentionBiLSTM(**model_kwargs)
    if model_kwargs['model_summary']:
        print(model)
        n_params = baselinemodels.get_n_params(model)
        n_train = baselinemodels.get_trainable_params(model)
        print(n_params, 'parameters in model.')
        #print(n_train, 'Trainable parameters in model.')
    return model
