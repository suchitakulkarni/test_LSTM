# Conservative quantile loss (penalizes under-prediction more)
def conservative_quantile_loss(preds, target, quantiles, conservatism_factor=1.5):
    """
    Conservative quantile loss that penalizes under-prediction more heavily
    conservatism_factor > 1 makes the model more conservative (predict lower RUL)
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        
        # Standard quantile loss
        standard_loss = torch.max((q - 1) * errors, q * errors)
        
        # Apply conservatism: penalize positive errors (under-prediction) more
        if conservatism_factor != 1.0:
            # Positive errors (under-prediction) get extra penalty
            penalty = torch.where(errors > 0,
                                errors * (conservatism_factor - 1.0) * q,
                                torch.zeros_like(errors))
            conservative_loss = standard_loss + penalty
        else:
            conservative_loss = standard_loss
            
        losses.append(conservative_loss)
    
    total_loss = torch.mean(torch.stack(losses, dim=1).sum(dim=1))
    return total_loss
