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
 * Initializer that generates a truncated normal distribution.
 * 
 * @author Jim Clarke
 */
public class TruncatedNormal<U extends TType> extends Initializer<U> {
    public final static String MEAN_KEY = "mean";
    public final static String STDDEV_KEY = "stddev";
    public final static String SEED_KEY = "seed";
    public final static double MEAN_DEFAULT = 0.0;
    public final static double STDDEV_DEFAULT = 0.05;
    

    private final double mean ;
    private final double stddev;
    private final Long seed;
    
    /**
     * create a TruncatedNormal Initializer
     * 
     * @param tf  the TensorFlow Ops
     */
    public TruncatedNormal(Ops tf) {
        this(tf, MEAN_DEFAULT, STDDEV_DEFAULT, null);
    }
    
    /**
     * create a TruncatedNormal Initializer
     * 
     * @param tf the TensorFlow Ops
     * @param mean Mean of the random values to generate.
     * @param stddev Standard deviation of the random values to generate.
     */
    public TruncatedNormal(Ops tf, double mean, double stddev ) {
        this(tf, mean, mean, null);
    }
    
    /**
     * create a TruncatedNormal Initializer
     * 
     * @param tf the TensorFlow Ops
     * @param mean Mean of the random values to generate.
     * @param stddev Standard deviation of the random values to generate.
     * @param seed Used to create random seeds
     */
    public TruncatedNormal(Ops tf, double mean, double stddev,Long seed ) {
        super(tf);
        this.mean = mean;
        this.stddev = stddev;
        this.seed = seed;
        
    }
    
    /**
     * create a TruncatedNormal Initializer
     * 
     * @param tf the TensorFlow Ops
     * @param config  the settings to initialize this initializer
     */
    public TruncatedNormal(Ops tf, Map<String, Object> config) {
        super(tf, config);
        this.mean = (double)config.getOrDefault(MEAN_KEY, MEAN_DEFAULT);
        this.stddev =  (double)config.getOrDefault(STDDEV_KEY, STDDEV_DEFAULT);
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
        long lseed = this.seed == null? 0L : this.seed.longValue();
        long[] seeds = { lseed, 0L };
        Operand distOp = tf.random.statelessTruncatedNormal(dims, tf.constant(seeds), (DataType)dtype);
        return tf.math.add(
                tf.math.mul(distOp, tf.dtypes.cast(tf.constant(stddev), dtype)),
                tf.dtypes.cast(tf.constant(mean), dtype));
    }
    
}
