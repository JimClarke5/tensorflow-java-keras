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
import org.tensorflow.keras.utils.TypeUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.op.random.RandomUniformInt;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TType;

/**
 *
 * @author Jim Clarke
 */
public class RandomUniform <U extends TType> extends Initializer<U> {
    public final static String MINVAL_KEY = "minval";
    public final static String MAXVAL_KEY = "maxval";
    public final static String SEED_KEY = "seed";
    public final static double MINVAL_DEFAULT = 0.05;
    public final static double MAXVAL_DEFAULT = 0.05;
    
    private final Double minval;
    private final Double maxval;
    private final Long seed;
    
    public RandomUniform() {
        this(MINVAL_DEFAULT, MAXVAL_DEFAULT, null);
    }
    
    public RandomUniform(double minval, double maxval) {
       this(minval, maxval, null);
    }
    public RandomUniform(double minval, double maxval, Long seed) {
        super();
        this.minval = minval;
        this.maxval = maxval;
        this.seed = seed;
    }
    
    public RandomUniform(Map<String, Object> config) {
        super(config);
        this.minval = (double)config.getOrDefault(MINVAL_KEY, MINVAL_DEFAULT);
        this.maxval = (double)config.getOrDefault(MAXVAL_KEY, MAXVAL_DEFAULT);
        this.seed = (Long)config.getOrDefault(SEED_KEY, null);
    }
    
    @Override
    public Map<String, Object> getConfig() {
        Map<String, Object> config = super.getConfig();
        config.put(MINVAL_KEY, minval);
        config.put(MAXVAL_KEY, maxval);
        config.put(SEED_KEY, seed);
        return config;
    }

    @Override
    public Operand<U> call(Ops tf, Operand<TInt64> dims, DataType<U> dtype) {
        double range = this.maxval - this.minval;
        double mean =  range/ 2.0;
        Operand distOp;
        if(TypeUtils.isInteger(dtype)) {
            if(this.maxval == null) {
                throw new IllegalArgumentException("Must specify maxval for integer dtype " + dtype.name());
            }
            RandomUniformInt.Options options = RandomUniformInt.seed(this.seed);
            distOp = tf.random.randomUniformInt(dims, 
                    tf.dtypes.cast(tf.constant(this.minval), (DataType)dtype),
                    tf.dtypes.cast(tf.constant(this.maxval), (DataType)dtype),
                    options); 
        }else {
            long lseed = this.seed == null? 0L : this.seed.longValue();
            long[] seeds = { lseed, 0L };
            distOp = tf.random.statelessRandomUniform(dims, tf.constant(seeds), (DataType)dtype);
            if(this.minval == 0) {
                if(this.minval != 1.0) {
                    distOp = tf.math.mul(distOp, tf.dtypes.cast(tf.constant(this.maxval), dtype));
                }
            }else {
                distOp = tf.math.mul(distOp, tf.dtypes.cast(tf.constant(this.maxval-this.minval), dtype));
                distOp = tf.math.add(distOp, tf.dtypes.cast(tf.constant(this.minval), dtype));
            }
        }
        return distOp;
    }
}
