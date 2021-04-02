# batch_idx = np.random.randint(0, data.samples // data.batch_size)

# imgs, labels = data[batch_idx]
# print(imgs.shape)

# start = batch_idx * data.batch_size

# end =  start + data.batch_size if start >= 0 else None
# indices = data.index_array[start:end]
# filepaths = [data.filepaths[i] for i in indices]

# probs, preds = predict(model_exp, imgs)
# probs = probs.reshape((probs.shape[0],))
# preds = preds.reshape((preds.shape[0],))

# conv_idx = [0, 3]
# conv_layers = [layer for layer in model_exp.layers if 'conv' in layer.name]
# selected_layers = [layer for index, layer in enumerate(conv_layers) if index in conv_idx]
# activation_model = Model(inputs=model_exp.inputs, outputs=[layer.output for layer in selected_layers])


# class_mapping = {value: key for key, value in data.class_indices.items()}
# plots = {}
# for filepath, img, label, prob, pred in zip(filepaths, imgs, labels, probs, preds):
#     plot_files = visualize_conv_layers(activation_model, img, conv_idx)
#     entry = {
#         'true_label': class_mapping[int(label)],
#         'pred_label': class_mapping[pred],
#         'probability': (1 - prob if pred == 0 else prob) * 100,
#         'plot_filepaths': plot_files,
#     }
#     plots[filepath] = entry
