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
import java.util.Map;
import org.tensorflow.Graph;
import static org.tensorflow.keras.optimizers.OptimizerInterface.NAME_KEY;

/**
 * RMSProp Optimizer that implements the RMSProp algorithm.
 *
 * @author Jim Clarke
 */
public class RMSProp extends org.tensorflow.framework.optimizers.RMSProp implements OptimizerInterface  {

    public static final String LEARNING_RATE_KEY = "learning_rate";
    public static final String DECAY_KEY = "decay";
    public static final String MOMENTUM_KEY = "momentum";
    public static final String EPSILON_KEY = "epsilon";
    public static final String CENTERED_KEY = "centered";
    
     
    public static final float LEARNING_RATE_DEFAULT = 0.01F;
    public static final float DECAY_DEFAULT = 0.9F;
    public static final float MOMENTUM_DEFAULT  = 0.0F;
    public static final float EPSILON_DEFAULT = 1e-07F;
    public static final boolean CENTERED_DEFAULT = false;
    
    private Map<String, Object> config = new HashMap<>();
     
    /**
     * create an GradientDescent
     * @param graph
     */
    public RMSProp(Graph graph) {
        this(graph, LEARNING_RATE_DEFAULT, DECAY_DEFAULT,  MOMENTUM_DEFAULT, 
                EPSILON_DEFAULT, CENTERED_DEFAULT);
    }
    
    public RMSProp(Graph graph, float learningRate) {
        this(graph, learningRate, DECAY_DEFAULT,  MOMENTUM_DEFAULT, 
                EPSILON_DEFAULT, CENTERED_DEFAULT);
    }
    
    public RMSProp(Graph graph, String name, float learningRate) {
        this(graph, name, learningRate, DECAY_DEFAULT,  MOMENTUM_DEFAULT, 
                EPSILON_DEFAULT, CENTERED_DEFAULT);
    }
    
     public RMSProp(Graph graph, float learningRate, float decay, float momentum, 
             float epsilon, boolean centered) {
         super(graph, learningRate, decay, momentum,  epsilon,  centered);
         initConfig(learningRate, decay, momentum,  epsilon,  centered);
     }
     
     public RMSProp(Graph graph, String name,  float learningRate, float decay, 
             float momentum,  float epsilon, boolean centered) {
         super(graph, name, learningRate, decay, momentum,  epsilon,  centered);
         initConfig(learningRate, decay, momentum,  epsilon,  centered);
     }

    /**
     * create an Adam
     *
     * @param graph
     * @param config a config object to initialize
     */
    public static  RMSProp create(Graph graph, Map<String, Object> config) {
        
        String name = (String)config.get(NAME_KEY);
        float learningRate = (float)config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        float decay = (float)config.getOrDefault(DECAY_KEY, DECAY_DEFAULT);
        float momentum = (float)config.getOrDefault(MOMENTUM_KEY, MOMENTUM_DEFAULT);
        float epsilon = (float)config.getOrDefault(EPSILON_KEY, EPSILON_DEFAULT);
        boolean centered = (boolean)config.getOrDefault(CENTERED_KEY, CENTERED_DEFAULT);
        if(name == null) 
            return new RMSProp(graph, learningRate, decay, momentum,  epsilon,  centered );
        else
            return new RMSProp(graph, name, learningRate, decay, momentum,  epsilon,  centered);
        
    }
    
    private void initConfig(float learningRate, float decay,  float momentum,
            float epsilon, boolean centered) {
        config.put(NAME_KEY, this.getOptimizerName());
        config.put(LEARNING_RATE_KEY, learningRate);
        config.put(DECAY_KEY, decay);
        config.put(MOMENTUM_KEY, momentum);
        config.put(EPSILON_KEY, epsilon);
        config.put(CENTERED_KEY, centered);
    }
    
     @Override
    public Map<String, Object> getConfig() {
        return config;
    }
}
