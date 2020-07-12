/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.activations;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

/**
 * Rectified Linear Unit(ReLU) activation.
 *
 * @param <T> the data type of the result
 */
public class ReLU<T extends TType> extends Activation<T> {

    private static final double ALPHA_DEFAULT = 0.0;
    private static final Double MAX_VALUE_DEFAULT = null;
    private static final double THRESHOLD_DEFAULT = 0.0;

    private final double alpha;
    private final Double max_value;
    private final double threshold;

    /**
     * Creates a new ReLU with alpha=0.0, max_value=null, threshold=0.0
     *
     * @param tf the TensorFlow Ops
     */
    public ReLU(Ops tf) {
        this(tf, ALPHA_DEFAULT, MAX_VALUE_DEFAULT, THRESHOLD_DEFAULT);
    }

    /**
     * Creates a new ReLU
     *
     * @param tf the TensorFlow Ops
     * @param alpha governs the slope for values lower than the threshold.
     * @param max_value sets the saturation threshold (the largest value the
     * function will return).
     * @param threshold the threshold value of the activation function below
     * which values will be damped or set to zero.
     */
    public ReLU(Ops tf, double alpha, Double max_value, double threshold) {
        super(tf);
        this.alpha = alpha;
        this.max_value = max_value;
        this.threshold = threshold;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand<T> call(Operand<T> input) {
        Operand negative_part = null;
        if (alpha != 0) {
            if (max_value == null && threshold == 0) {
                //return leaky_relu(input, alpha)
            }
            if (threshold != 0) {
                negative_part = tf.nn.relu(tf.math.add(tf.math.neg(input),
                        tf.dtypes.cast(tf.constant(threshold), input.asTensor().dataType())));
            } else {
                negative_part = tf.nn.relu(tf.math.neg(input));
            }
        }
        boolean clip_max = max_value != null;
        if (threshold != 0) {
            // computes input for input > threshold else 0
            // TODO
            //input = input * math_ops.cast(math_ops.greater(input, threshold), floatx())
            Operand op = tf.math.greater((Operand) input, tf.dtypes.cast(tf.constant(threshold), (DataType) input.asTensor().dataType()));
            input = tf.math.mul(input, op);
        } else if (max_value != null && max_value == 6) {
            // if no threshold, then can use nn.relu6 native TF op for performance
            input = tf.nn.relu6((Operand) input);
            clip_max = false;
        } else {
            input = tf.nn.relu(input);
        }
        if (clip_max) {
            Operand lmax_value = tf.dtypes.cast(tf.constant(max_value), input.asTensor().dataType());
            Operand zero = tf.dtypes.cast(tf.constant(0), input.asTensor().dataType());
            input = tf.clipByValue(input, zero, lmax_value);
        }

        if (alpha != 0.) {
            Operand lalpha = tf.dtypes.cast(tf.constant(alpha), input.asTensor().dataType());
            input = tf.math.sub(input, tf.math.mul(lalpha, negative_part));
        }
        return input;
    }

}
