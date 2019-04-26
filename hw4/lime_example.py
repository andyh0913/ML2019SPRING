images = [x_train[i] for i in path_list]
    y = [y_train[i] for i in path_list]
    images = np.array(images)
    y = np.array(y)
    explainer = lime_image.LimeImageExplainer()
    predict_ = lambda x : np.squeeze(model.predict(x[:, :, :, 0].reshape(-1, 48, 48, 1)))
    for i in range(7):
        image = [images[i]] * 3
        image = np.concatenate(image, axis = 2)
        np.random.seed(16)
        explanation = explainer.explain_instance(image, predict_, labels=(i, ), top_labels=None, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=1000, hide_rest=True)
        plt.imsave(out_path + 'fig3_' + str(i) + '.jpg', mark_boundaries(temp / 2 + 0.5, mask))