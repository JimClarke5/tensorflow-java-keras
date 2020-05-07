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
import org.tensorflow.Output;
import org.tensorflow.keras.utils.ShapeUtils;
import org.tensorflow.keras.utils.TypeUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.op.linalg.Qr;
import org.tensorflow.op.random.RandomStandardNormal;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TType;

/**
 *  Initializer that generates an orthogonal matrix.
 * @author Jim Clarke
 */
public class Orthogonal<U extends TType> extends Initializer<U> {
    
    public static final String GAIN_KEY = "gain";
    public static final double GAIN_DEFAULT = 1.0;
    public final static String SEED_KEY = "seed";
    
    private final double gain;
    private final Long seed;
    
    /**
     * Creates an Orthogonal Initializer
     */
    public Orthogonal() {
        this(GAIN_DEFAULT, null);
    }
    
    /**
     * Creates an Orthogonal Initializer
     * @param gain the gain to be applied to the Matrix
     */
    public Orthogonal(double gain) {
        this(gain, null);
    }
    
    /**
     * Creates an Orthogonal Initializer
     * @param gain the gain to be applied to the Matrix
     * @param seed the seed for random number generation
     */
    public Orthogonal(double gain, Long seed) {
        super();
        this.gain = gain;
        this.seed = seed;
    }
    
     /**
     * create a Orthogonal initializer
     * @param config the config object used to initialize the Initializer values
     */
    public Orthogonal(Map<String, Object> config) {
        super(config);
        this.gain = (double)config.getOrDefault(GAIN_KEY, GAIN_DEFAULT);
        this.seed = (Long)config.getOrDefault(SEED_KEY, null);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Map<String, Object> getConfig() {
        Map<String, Object> config = super.getConfig();
        config.put(GAIN_KEY, gain);
        config.put(SEED_KEY, seed);
        return config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand<U> call(Ops tf, Operand<TInt64> dims, DataType<U> dtype) {
        assert TypeUtils.isFloating(dtype) : String.format("Expected floating point type, got %s.",dtype);
        Shape dimsShape = ShapeUtils.getShape(dims);
        assert(dimsShape.numDimensions() >= 2): "The tensor to initialize must be at least two-dimensional";
        long num_rows = 1;
        int i = 0;
        for(; i < dimsShape.numDimensions()-1; i++)
            num_rows *= dimsShape.size(i);
        long num_cols = dimsShape.size(i);
        Shape flat_shape = Shape.of(Math.max(num_rows, num_cols), Math.min(num_rows, num_cols));
        long lseed = this.seed == null? 0L : this.seed;
        long[]seeds = {lseed, 0};
        Operand op = tf.random.statelessRandomNormal(tf.constant(flat_shape), tf.constant(seeds), (DataType)dtype);
        
        Qr.Options qrOptions =  Qr.fullMatrices(false);
        Qr qrOp = tf.linalg.qr(op, qrOptions);
        Output qo = qrOp.q();
        Output ro = qrOp.r();
        Operand diagOp =  tf.linalg.matrixDiagPart(
                ro, 
                tf.constant(0), 
                tf.dtypes.cast(tf.constant(0), dtype));
        Operand qop = tf.math.mul(qo, tf.math.sign(diagOp));
        if(num_rows < num_cols) 
            qop = tf.linalg.transpose(qop, null);
        
        //TODO, do we need to reshape?
        return tf.math.mul(qop, tf.dtypes.cast(tf.constant(this.gain), dtype));
    }
    
}
