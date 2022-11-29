from tensorflow import keras


def run_experiment(model, x_train, y_train, learning_rate, loss, num_epochs, batch_size, optimizer):
    # Compile the model.
    model.compile(
        optimizer=optimizer(learning_rate),
        loss = loss(from_logits=True),
        metrics=['accuracy'],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        patience=2
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping, reduce_lr],
    )
    return history