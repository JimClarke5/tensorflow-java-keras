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
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TType;

/**
 *
 * @author Jim Clarke
 */
public class VarianceScaling <U extends TType> extends Initializer<U> {
    
    public static final String SCALE_KEY = "scale";
    public static final String MODE_KEY = "mode";
    public static final String DISTRIBUTION_KEY = "distribution";
    public static final String SEED_KEY = "seed";
    
    public static final double SCALE_DEFAULT = 1.0;
    public static final String MODE_DEFAULT = "fan_in";
    public static final String DISTRIBUTION_DEFAULT = "truncated_normal";
    
    private double scale;
    private Mode mode;
    private Distribution distribution;
    private Long seed;
    
    
    public VarianceScaling() {
        this(SCALE_DEFAULT, MODE_DEFAULT, DISTRIBUTION_DEFAULT, null);
    }
    
    public VarianceScaling(long seed) {
        this(SCALE_DEFAULT, MODE_DEFAULT, DISTRIBUTION_DEFAULT, seed);
    }
    
     public VarianceScaling(double scale, String mode, String distribution, Long seed) {
         super();
         assert(scale > 0.0);
         this.scale = scale;
         this.mode = Mode.valueOf(mode);
         this.distribution = Distribution.valueOf(distribution);
         this.seed = seed;
         
     }
    
    public VarianceScaling(Map<String, Object> config) {
        super(config);
        this.scale = (double)config.getOrDefault(SCALE_KEY, SCALE_DEFAULT);
        this.mode =  Mode.valueOf((String)config.getOrDefault(MODE_KEY, MODE_DEFAULT));
        this.distribution = Distribution.valueOf((String)config.getOrDefault(DISTRIBUTION_KEY, DISTRIBUTION_DEFAULT));
        this.seed = (Long)config.getOrDefault(SEED_KEY, null);
    }
    
    @Override
    public Map<String, Object> getConfig() {
        Map<String, Object> config = super.getConfig();
        config.put(SCALE_KEY, scale);
        config.put(MODE_KEY, mode.name());
        config.put(DISTRIBUTION_KEY, distribution.name());
        config.put(SEED_KEY, seed);
        return config;
    }

    @Override
    public Operand<U> call(Ops tf, Operand<TInt64> dims, DataType<U> dtype) {
        assert(TypeUtils.isFloating(dtype));
        Shape shape = ShapeUtils.getShape(dims);
        double lscale = this.scale;
        double[] fans /* fan_in, fan_out */ = _compute_fans(shape);
        switch(mode) {
            case fan_in:
                lscale /= Math.max(1., fans[0]);
                break;
            case fan_out:
                lscale /= Math.max(1., fans[1]);
                break;
            case fan_avg:
                lscale /= Math.max(1., (fans[0] + fans[1]) / 2.);
                break;
        }
        Operand<U> distOp;
        Operand<U> mulOp = null;
        double stddev;
        long lseed = this.seed == null? 0L : this.seed;
        long[] seeds = { lseed, 0L };
        switch(distribution) {
            case truncated_normal:
                distOp = tf.random.statelessTruncatedNormal(dims, tf.constant(seeds), (DataType)dtype);
                // constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
                stddev = Math.sqrt(lscale) / .87962566103423978;
                mulOp = tf.math.mul(distOp, tf.dtypes.cast(tf.constant(stddev), dtype));
                break;
            case untruncated_normal:
                distOp = tf.random.statelessRandomNormal(dims, tf.constant(seeds), (DataType)dtype);
                stddev = Math.sqrt(lscale);
                mulOp = tf.math.mul(distOp, tf.dtypes.cast(tf.constant(stddev), dtype));
                break;
            case uniform:      
                distOp = tf.random.statelessRandomUniform(dims, tf.constant(seeds), (DataType)dtype);
                stddev = Math.sqrt(3.0 * lscale);
                mulOp = tf.math.mul(distOp, tf.dtypes.cast(tf.constant(stddev), dtype));
                break;

        }
        return mulOp;
        
    }
    
    private double[] _compute_fans(Shape shape) {
        double fan_in = 0.0;
        double fan_out = 1.0;
        long[] dims = shape.asArray();
        if(dims.length < 1) {
           fan_in = fan_out = 1;
        }else if(dims.length == 1) {
            fan_in = fan_out = dims[0];
        }else if(dims.length == 2) {
            fan_in = dims[0];
            fan_out = dims[1];
        }else {
            double receptive_field_size = 1.;
            for(int i = dims.length-2; i >= 0; i--) {
                receptive_field_size *= dims[i];
            }
            fan_in = dims[dims.length-2] * receptive_field_size;
            fan_out = dims[dims.length-1] * receptive_field_size;
        }
        
        return new double[] { fan_in, fan_out};
    }
    
    public static enum Mode {
         fan_in, fan_out, fan_avg;
    }
    
    public static enum Distribution {
         truncated_normal, untruncated_normal, uniform;
    }

    
}
