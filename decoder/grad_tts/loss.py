def compute_loss(model, batch):
    
    mel, ling_rep, pros_rep, spk_emb, length, max_len = batch
    
    mel = mel.transpose(1,2)
    ling_rep = ling_rep.transpose(1,2)
    pros_rep = pros_rep.transpose(1,2)
    loss, losses = model(ling_rep,
                                        length,
                                        mel,
                                        length,
                                        spk_emb,
                                        pros_rep
                                    )
    return loss, losses 
