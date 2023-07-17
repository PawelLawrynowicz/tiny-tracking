
import tensorflow as tf

# Load model without signatures
model = tf.saved_model.load('./ssd_mobilenet_v2_coco_2018_03_29/saved_model')

# Add an input signature


@tf.function(input_signature=[tf.TensorSpec(shape=(1, 320, 320, 3), dtype=tf.uint8, name="image_tensor")])
def predict_fn(input):
    output = model(input)
    return output


predict_concrete_fn = predict_fn.get_concrete_function()

tf.saved_model.save(model,
                    'ssd_mobilenet_v2_coco_2018_03_29',
                    signatures={'predict': predict_concrete_fn}
                    )

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(
    './ssd_mobilenet_v2_coco_2018_03_29')  # path to the SavedModel directory
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
