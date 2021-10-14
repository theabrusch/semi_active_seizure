from src.models import baselinemodels

def get_model(model_kwargs):
    if model_kwargs['model'] == 'BaselineCNN':
        model = baselinemodels.BaselineCNN(**model_kwargs)
    if model_kwargs['model_summary']:
        print(model)
        n_params = baselinemodels.get_n_params(model)
        print(n_params, 'parameters in model.')
    return model
