/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the );
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an  BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.optimizers;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.op.Op;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.train.ApplyFtrl;
import org.tensorflow.types.family.TType;

/**
 * Ftrl Optimizer that implements the FTRL algorithm.
 *
 * @author Jim Clarke
 */
public class Ftrl extends org.tensorflow.framework.optimizers.Optimizer implements OptimizerInterface  {
    public static final String LEARNING_RATE_KEY = "learning_rate";
    public static final String LEARNING_RATE_POWER_KEY = "learning_rate_power";
    public static final String INITIAL_ACCUM_VALUE_KEY = "initial_accumulator_value";
    public static final String L1STRENGTH_KEY = "l1_regularization_strength";
    public static final String L2STRENGTH_KEY = "l2_regularization_strength";
    public static final String L2_SHRINKAGE_REGULARIZATION_STRENGTH_KEY = "l2_shrinkage_regularization_strength";
    
    public static final float LEARNING_RATE_DEFAULT = 0.001F;
    public static final float LEARNING_RATE_POWER_DEFAULT = -0.5F;
    public static final float INITIAL_ACCUM_VALUE_DEFAULT = 0.1F;
    public static final float L1STRENGTH_DEFAULT = 0.0F;
    public static final float L2STRENGTH_DEFAULT = 0.0F;
    public static final float L2_SHRINKAGE_REGULARIZATION_STRENGTH_DEFAULT = 0.0F;
    
    public static final String ACCUMULATOR = "gradient_accumulator";
    public static final String LINEAR_ACCUMULATOR = "linear_accumulator";
    
    private final String name;
    private final float learningRate;
    private final float learningRatePower;
    private final float initialAccumulatorValue;
    private final float l1RegularizationStrength;
    private final float l2RegularizationStrength;
    private final float l2ShrinkageRegularizationStrength;
    
    private Map<String, Object> config = new HashMap<>();
    
    private boolean useLocking = true;
    
    
      /**
     * create an Ftrl
     * @param graph
     */
    public Ftrl(Graph graph) {
       this(graph, LEARNING_RATE_DEFAULT, LEARNING_RATE_POWER_DEFAULT,
               INITIAL_ACCUM_VALUE_DEFAULT, L1STRENGTH_DEFAULT, L2STRENGTH_DEFAULT,
               L2_SHRINKAGE_REGULARIZATION_STRENGTH_DEFAULT);
    }
    
    public Ftrl(Graph graph, String name) {
         this(graph, name, LEARNING_RATE_DEFAULT, LEARNING_RATE_POWER_DEFAULT,
               INITIAL_ACCUM_VALUE_DEFAULT, L1STRENGTH_DEFAULT, L2STRENGTH_DEFAULT,
               L2_SHRINKAGE_REGULARIZATION_STRENGTH_DEFAULT);
    }
    
    public Ftrl(Graph graph, float learningRate ) {
       this(graph, learningRate, LEARNING_RATE_POWER_DEFAULT,
               INITIAL_ACCUM_VALUE_DEFAULT, L1STRENGTH_DEFAULT, L2STRENGTH_DEFAULT,
               L2_SHRINKAGE_REGULARIZATION_STRENGTH_DEFAULT);
    }
    
    public Ftrl(Graph graph, String name,  float learningRate) {
         this(graph, name, learningRate, LEARNING_RATE_POWER_DEFAULT,
               INITIAL_ACCUM_VALUE_DEFAULT, L1STRENGTH_DEFAULT, L2STRENGTH_DEFAULT,
               L2_SHRINKAGE_REGULARIZATION_STRENGTH_DEFAULT);
    }
    
    public Ftrl(Graph graph, float learningRate, float learningRatePower,
            float initialAccumulatorValue, float l1Strength, float l2Strength,
           float  l2ShrinkageRegularizationStrength) {
        super(graph);
        this.name = getOptimizerName();
        this.learningRate = learningRate;
        this.learningRatePower = learningRatePower;
        this.initialAccumulatorValue = initialAccumulatorValue;
        this.l1RegularizationStrength = l1Strength;
        this.l2RegularizationStrength = l2Strength;
        this.l2ShrinkageRegularizationStrength = l2ShrinkageRegularizationStrength;
        validateParams();
        initConfig();
    }
    
    public Ftrl(Graph graph, String name, float learningRate, float learningRatePower,
            float initialAccumulatorValue, float l1Strength, float l2Strength,
           float  l2ShrinkageRegularizationStrength) {
        super(graph, name);
        this.name = name;
        this.learningRate = learningRate;
        this.learningRatePower = learningRatePower;
        this.initialAccumulatorValue = initialAccumulatorValue;
        this.l1RegularizationStrength = l1Strength;
        this.l2RegularizationStrength = l2Strength;
        this.l2ShrinkageRegularizationStrength = l2ShrinkageRegularizationStrength;
        validateParams();
        initConfig();
    }

    /**
     * create an Adam
     *
     * @param graph
     * @param config a config object to initialize
     * @return 
     */
    public static  Ftrl create(Graph graph, Map<String, Object> config) {
        String name = (String)config.get(NAME_KEY);
        float learningRate = (float)config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        float learningRatePower =  (float)config.getOrDefault(LEARNING_RATE_POWER_KEY, LEARNING_RATE_POWER_DEFAULT);
        float initialAccumulatorValue =  (float)config.getOrDefault(INITIAL_ACCUM_VALUE_KEY, INITIAL_ACCUM_VALUE_DEFAULT);
        float l1RegularizationStrength =  (float)config.getOrDefault(L1STRENGTH_KEY, L1STRENGTH_DEFAULT);
        float l2RegularizationStrength =  (float)config.getOrDefault(L2STRENGTH_KEY, L2STRENGTH_DEFAULT);
        float l2ShrinkageRegularizationStrength =  
                (float)config.getOrDefault(L2_SHRINKAGE_REGULARIZATION_STRENGTH_KEY, L2_SHRINKAGE_REGULARIZATION_STRENGTH_DEFAULT);
        
        if(name == null) 
            return new Ftrl(graph, learningRate, learningRatePower, initialAccumulatorValue, 
                    l1RegularizationStrength, l2RegularizationStrength,
                    l2ShrinkageRegularizationStrength);
        else
             return new Ftrl(graph, name, learningRate, learningRatePower, initialAccumulatorValue, 
                    l1RegularizationStrength, l2RegularizationStrength,
                    l2ShrinkageRegularizationStrength);
    }
    
    protected void initConfig() {
        config.put(NAME_KEY, this.name);
        config.put(LEARNING_RATE_KEY, learningRate);
        config.put(LEARNING_RATE_POWER_KEY, learningRatePower);
        config.put(INITIAL_ACCUM_VALUE_KEY, initialAccumulatorValue);
        config.put(L1STRENGTH_KEY, l1RegularizationStrength);
        config.put(L2STRENGTH_KEY, l2RegularizationStrength);
        config.put(L2_SHRINKAGE_REGULARIZATION_STRENGTH_KEY, l2ShrinkageRegularizationStrength);
    }
    
    private void validateParams() {
        if(this.initialAccumulatorValue < 0.0F)
            throw new IllegalArgumentException(
                    String.format("initialAccumulatorValue %f needs to be positive or zero", this.initialAccumulatorValue));
        if(this.learningRatePower > 0.0F)
            throw new IllegalArgumentException(
                    String.format("learningRatePower %f needs to be negative or zero", this.learningRatePower));
        if(this.l1RegularizationStrength < 0.0F)
            throw new IllegalArgumentException(
                    String.format("'l1RegularizationStrength %f needs to be positive or zero", this.l1RegularizationStrength));
        if(this.l2RegularizationStrength < 0.0F)
            throw new IllegalArgumentException(
                    String.format("'l2RegularizationStrength %f needs to be positive or zero", this.l2RegularizationStrength));
        if(this.l2ShrinkageRegularizationStrength < 0.0F)
            throw new IllegalArgumentException(
                    String.format("'l2ShrinkageRegularizationStrength %f needs to be positive or zero", this.l2RegularizationStrength));

    
    }

    
    @Override
    protected void createSlots(List<Output<? extends TType>> variables) {
      for (Output<? extends TType> v : variables) {
        createFtrlSlot(v);
      }
    }
    
    private <T extends TType> void createFtrlSlot(Output<T> v) {
        Operand<T> initializer = tf
            .fill(tf.shape(v), tf.dtypes.cast(tf.constant(initialAccumulatorValue), v.dataType()));
        createSlot(v.asOutput(), ACCUMULATOR, initializer);
        Operand<T> linearInitializer = tf.fill(tf.shape(v),
            tf.dtypes.cast(tf.constant(0.0f), v.dataType()));
        createSlot(v.asOutput(), LINEAR_ACCUMULATOR, linearInitializer);
  }
  
    @Override
    protected <T extends TType> Op applyDense(Output<T> gradient, Output<T> variable) {
        Variable<T> accumSlot = getSlot(variable, ACCUMULATOR).get();
        Variable<T> linearSlot = getSlot(variable, LINEAR_ACCUMULATOR).get();
        ApplyFtrl.Options options = ApplyFtrl.useLocking(useLocking);
        return this.tf.train.applyFtrl(
                variable,
                accumSlot, //accum
                linearSlot, //linear
                gradient, //gradient
                tf.dtypes.cast(tf.constant(learningRate), gradient.dataType()), // lr
                tf.dtypes.cast(tf.constant(l1RegularizationStrength), gradient.dataType()), //l1
                tf.dtypes.cast(tf.constant(l2RegularizationStrength), gradient.dataType()),  // l2
                tf.dtypes.cast(tf.constant(l2ShrinkageRegularizationStrength), gradient.dataType()), // l2Shrinkage
                tf.dtypes.cast(tf.constant(learningRatePower), gradient.dataType()), //lrPower
                options);
   
    }

    @Override
    public String getOptimizerName() {
        return "Ftrl";
    }
    
    @Override
    public Map<String, Object> getConfig() {
        return config;
    }

}
