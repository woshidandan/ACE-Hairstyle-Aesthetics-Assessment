import torch
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def validate(model, dataloader, regression_criterion, angle_criterion):
    model.eval()
    val_loss = 0.0
    regression_loss_total = 0.0
    angle_loss_total = 0.0

    with torch.no_grad():
        for hair_images, score, angle_labels in tqdm(dataloader, desc="Validating"):
            hair_images = hair_images.to(device)
            face_image = hair_images[:, 0, :, :, :]
            score = score.to(device)
            angle_labels = angle_labels.to(device)

            outputs, angle_predictions = model(hair_images, face_image)
            regression_loss = regression_criterion(outputs, score)
            angle_loss = angle_criterion(angle_predictions.view(-1, 12), angle_labels.view(-1))

            total_loss = regression_loss + 0.5 * angle_loss
            val_loss += total_loss.item()
            regression_loss_total += regression_loss.item()
            angle_loss_total += angle_loss.item()

    avg_val_loss = val_loss / len(dataloader)
    avg_regression_loss = regression_loss_total / len(dataloader)
    avg_angle_loss = angle_loss_total / len(dataloader)

    print(f"Validation - Regression Loss: {avg_regression_loss:.4f}, Angle Loss: {avg_angle_loss:.4f}, Total Loss: {avg_val_loss:.4f}")

    return avg_regression_loss


def train(
    model,
    train_loader,
    val_loader,
    regression_criterion,
    angle_criterion,
    optimizer,
    scheduler,
    writer,
    num_epochs,
    save_path,
    last_epoch_path="last_epoch.pth",
    update_centroids=True,
):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        regression_loss_total = 0.0
        angle_loss_total = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        if update_centroids and epoch % 10 == 0:
            with torch.no_grad():
                face_feats, hair_feats, age_feats, color_feats, skin_color_feats = [], [], [], [], []

                for hair_images, _, _ in train_loader:
                    face_image = hair_images[:, 0].to(device)

                    face_feats.append(model.face_encoder.forward_encoder(face_image).mean(dim=1))

                    hair_feats.append(model.multi_angle_extractor(hair_images.to(device))[0].mean(dim=1))

                    age_feats.append(face_feats[-1])

                    color_feats.append(hair_feats[-1])

                    skin_color_feats.append(face_feats[-1])

                model.rule_layer.update_centroids(
                    torch.cat(face_feats),
                    torch.cat(hair_feats),
                    torch.cat(age_feats),
                    torch.cat(color_feats),
                    torch.cat(skin_color_feats)
                )


        for hair_images, score, angle_labels in train_loader_tqdm:
            hair_images = hair_images.to(device)
            face_image = hair_images[:, 0, :, :, :]
            score = score.to(device)
            angle_labels = angle_labels.to(device)

            optimizer.zero_grad()

            outputs, angle_predictions = model(hair_images, face_image)

            regression_loss = regression_criterion(outputs, score)
            angle_loss = angle_criterion(angle_predictions.view(-1, 12), angle_labels.view(-1))
            rule_reg_loss = (torch.norm(model.rule_layer.rule_matrix_1, p=1) +
                            torch.norm(model.rule_layer.rule_matrix_2, p=1) +
                            torch.norm(model.rule_layer.rule_matrix_3, p=1)) * 0.1

            total_loss = regression_loss + 0.5 * angle_loss + rule_reg_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            regression_loss_total += regression_loss.item()
            angle_loss_total += angle_loss.item()

            train_loader_tqdm.set_postfix({
                "Train Loss": total_loss.item(),
                "Regression Loss": regression_loss.item(),
                "Angle Loss": angle_loss.item(),
                "Rule Reg Loss": rule_reg_loss.item()
            })

        avg_train_loss = running_loss / len(train_loader)
        avg_regression_loss = regression_loss_total / len(train_loader)
        avg_angle_loss = angle_loss_total / len(train_loader)

        writer.add_scalar('Loss/Training', avg_train_loss, epoch)
        writer.add_scalar('Loss/Regression', avg_regression_loss, epoch)
        writer.add_scalar('Loss/Angle', avg_angle_loss, epoch)

        avg_val_loss = validate(model, val_loader, regression_criterion, angle_criterion)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")

        if epoch + 1 == num_epochs and last_epoch_path is not None:
            torch.save(model.state_dict(), last_epoch_path)
            print(f"Last epoch model saved to {last_epoch_path}.")

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Regression Loss = {avg_regression_loss:.4f}, Angle Loss = {avg_angle_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
