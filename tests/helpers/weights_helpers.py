def get_weights(model):
    weights = []
    for param in model.parameters():
        param_weights = param.cpu().detach().numpy().copy()
        weights.append(param_weights)

    return weights


def check_weights_equal(state1, state2):
    for layer1, layer2 in zip(state1, state2):
        are_identical = (layer1 == layer2).all()
        if not are_identical:
            return False
    return True