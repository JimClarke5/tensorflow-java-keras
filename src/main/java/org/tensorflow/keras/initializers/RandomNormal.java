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
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TType;

/**
 *  Initializer that generates tensors with a normal distribution.
 * @author Jim Clarke
 */
public class RandomNormal<U extends TType> extends Initializer<U> {
    public final static String MEAN_KEY = "mean";
    public final static String STDDEV_KEY = "stddev";
    public final static String SEED_KEY = "seed";
    public final static double MEAN_DEFAULT = 0.0;
    public final static double STDDEV_DEFAULT = 1.0;
    
    private final double mean;
    private final double stddev;
    private final Long seed;
    
    /**
     * creates the RandomUniform initializer 
     * 
     * @param tf  the TensorFlow Ops
     */
    public RandomNormal(Ops tf) {
        this(tf, MEAN_DEFAULT, STDDEV_DEFAULT, null);
    }
    
    /**
     * creates the RandomUniform initializer 
     * 
     * @param tf the TensorFlow Ops
     * @param mean Mean of the random values to generate.
     */
    public RandomNormal(Ops tf, double mean) {
       this(tf, mean, STDDEV_DEFAULT, null);
    }
    
    /**
     * creates the RandomUniform initializer 
     * 
     * @param tf the TensorFlow Ops
     * @param mean Mean of the random values to generate.
     * @param stddev  Standard deviation of the random values to generate.
     */
    public RandomNormal(Ops tf, double mean, double stddev) {
       this(tf, mean, stddev, null);
    }
    
    /**
     * creates the RandomUniform initializer 
     * 
     * @param tf the TensorFlow Ops
     * @param mean Mean of the random values to generate.
     * @param stddev Standard deviation of the random values to generate.
     * @param seed  Used to create random seeds. 
     */
    public RandomNormal(Ops tf, double mean, double stddev, Long seed) {
        super(tf);
        this.mean = mean;
        this.stddev = stddev;
        this.seed = seed;
    }
    
    /**
     * creates the RandomUniform initializer 
     * 
     * @param tf the TensorFlow Ops
     * @param config the settings to initialize this initializer
     */
    public RandomNormal(Ops tf, Map<String, Object> config) {
        super(tf, config);
        this.mean = (double)config.getOrDefault(MEAN_KEY, MEAN_DEFAULT);
        this.stddev = (double)config.getOrDefault(STDDEV_KEY, STDDEV_DEFAULT);
        this.seed = (Long)config.getOrDefault(SEED_KEY, null);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Map<String, Object> getConfig() {
        Map<String, Object> config = super.getConfig();
        config.put(MEAN_KEY, mean);
        config.put(STDDEV_KEY, stddev);
        config.put(SEED_KEY, seed);
        return config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
   public Operand<U> call(Operand<TInt64> dims, DataType<U> dtype) {
        long lseed = this.seed == null ? 0L : this.seed;
        long[] seeds = { lseed, 0L };
        Operand distOp = tf.random.statelessRandomNormal(dims, tf.constant(seeds), (DataType)dtype);
        Operand<U> op = tf.math.mul(distOp, tf.dtypes.cast(tf.constant(this.stddev), dtype));
        return tf.math.add(op, tf.dtypes.cast(tf.constant(mean), dtype));
    }

    
}
