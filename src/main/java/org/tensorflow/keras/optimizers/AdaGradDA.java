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

import java.util.Map;
import org.tensorflow.Graph;

/**
 * Optimizer that implements the Adagrad Dual-Averaging algorithm.
 *
 * @author Jim Clarke
 */
public class AdaGradDA extends org.tensorflow.framework.optimizers.AdaGradDA implements OptimizerInterface  {
    public static final String LEARNING_RATE_KEY = "learning_rate";
    public static final String INITIAL_ACCUM_KEY = "accumulator";
    public static final String L1STRENGTH_KEY = "l1Strength";
    public static final String L2STRENGTH_KEY = "l2Strength";
    
    public static final float  LEARNING_RATE_DEFAULT = 0.001F;
    public static final float L1STRENGTH_DEFAULT = 0.0F;
    public static final float L2STRENGTH_DEFAULT = 0.0F;
  
    /**
     * create an AdagradDA
     * @param graph
     */
    public AdaGradDA(Graph graph) {
        this(graph, LEARNING_RATE_DEFAULT, LEARNING_RATE_DEFAULT, L1STRENGTH_DEFAULT, L2STRENGTH_DEFAULT);
    }
    
    public AdaGradDA(Graph graph, float learningRate) {
        this(graph, learningRate, LEARNING_RATE_DEFAULT, L1STRENGTH_DEFAULT, L2STRENGTH_DEFAULT);
    }
    
    public AdaGradDA(Graph graph, String name, float learningRate) {
        this(graph, name, learningRate, LEARNING_RATE_DEFAULT, L1STRENGTH_DEFAULT, L2STRENGTH_DEFAULT);
    }
    
    public AdaGradDA(Graph graph, float learningRate, float initialAccumulatorValue, float l1Strength,
      float l2Strength) {
        super(graph, learningRate, initialAccumulatorValue,l1Strength, l2Strength);
    }
    
    public AdaGradDA(Graph graph, String name, float learningRate, float initialAccumulatorValue, float l1Strength,
      float l2Strength) {
        super(graph, name, learningRate, initialAccumulatorValue,l1Strength, l2Strength);
    }

    /**
     * create an AdagradDA
     *
     * @param graph
     * @param config a config object to initialize
     */
    public static  AdaGradDA create(Graph graph, Map<String, Object> config) {
        return new AdaGradDA(graph);
    }
    
    private void initConfig() {
        config.put(NAME_KEY, this.getOptimizerName());
    }
    

}
