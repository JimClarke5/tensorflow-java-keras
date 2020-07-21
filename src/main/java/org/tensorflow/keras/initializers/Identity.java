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
package org.tensorflow.keras.initializers;

import java.util.Map;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.ShapeUtils;
import org.tensorflow.keras.utils.TypeUtils;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TType;

/**
 * Initializer that generates the identity matrix.
 * @param <U> The Type for the call operation
 * @author Jim Clarke
 */
public class Identity<U extends TType> extends Initializer<U> {
    public static final String GAIN_KEY = "gain";
    public static final double GAIN_DEFAULT = 1.0;
    
    private final double gain;
    
    /**
     * Creates an Initializer that generates the identity matrix.
     * 
     * @param tf the TensorFlow Ops
     */
    public Identity(Ops tf) {
        super(tf);
        this.gain = GAIN_DEFAULT;
    }
    
    /**
     * Creates an Initializer that generates the identity matrix.
     * 
     * @param tf the TensorFlow Ops
     * @param gain the gain to be applied to the (dentiy Matrix
     */
    public Identity(Ops tf, double gain) {
        super(tf);
        this.gain = gain;
    }
    
    /**
     *  Creates an Initializer that generates the identity matrix.
     * 
     * @param tf the TensorFlow Ops
     * @param config the config object used to initialize this Matrix
     */
    public Identity(Ops tf, Map<String, Object> config) {
        super(tf, config);
        this.gain = (double)config.getOrDefault(GAIN_KEY, GAIN_DEFAULT);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand<U> call(Operand<TInt64> dims, DataType<U> dtype) {
        assert(TypeUtils.isFloating(dtype));
        Shape shape = ShapeUtils.getShape(dims);
        assert(shape.numDimensions() == 2); // Only usable for generating 2D matrices.
        boolean isSquare = shape.size(0) == shape.size(1);
        long diag_size = Math.min( shape.size(0), shape.size(1));
        Shape diagShape = Shape.of(diag_size);
        
        Operand op;
        Operand zero = tf.dtypes.cast(tf.constant(0), dtype);
        Operand diag_ones = tf.fill(tf.constant(diagShape.asArray()), 
                                tf.dtypes.cast(tf.constant(1.0), dtype));
        if(isSquare) {
            op =  tf.linalg.matrixDiag(
                diag_ones, 
                tf.constant(0), // don't cast here, expecting TInt32
                tf.constant((int)shape.size(0)), 
                tf.constant((int)shape.size(1)), 
                zero);
        }else {
            Operand zero_matrix = tf.zeros(dims, dtype);
            op = tf.linalg.matrixSetDiag(zero_matrix, diag_ones, zero);
        }
        
        return tf.math.mul(op, tf.dtypes.cast(tf.constant(gain), dtype));
    }
    
}
