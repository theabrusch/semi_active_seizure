from src.models import baselinemodels

def get_model(model_kwargs):
    if model_kwargs['model'] == 'BaselineCNN':
        model = baselinemodels.BaselineCNN(**model_kwargs)
    if model_kwargs['model_summary']:
        print(model)
    return model
