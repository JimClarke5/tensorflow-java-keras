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
package org.tensorflow.keras.metrics;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;

/**
 *
 * @author jbclarke
 */
public class MeanTensorTest {
    
     private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    float epsilon = 1e-4F;
    
    public MeanTensorTest() {
    }
    
    @BeforeAll
    public static void setUpClass() {
    }
    
    @AfterAll
    public static void tearDownClass() {
    }
    
    @BeforeEach
    public void setUp() {
    }
    
    @AfterEach
    public void tearDown() {
    }

    @Test
    public void testConfig() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            
            MeanTensor instance = new MeanTensor(tf,"mean_by_element");
            session.run(tf.init());
            assertEquals("mean_by_element", instance.getName());
            assertEquals(TFloat32.DTYPE, instance.getDataType());
            assertEquals(0, instance.getVariables().size());
            
            try {
                instance.result();
            }catch(IllegalStateException expected) {
            }
            
            Op update = instance.updateState(tf.constant(new int[][]{{3}, {5}, {3}}));
            assertEquals(2, instance.getVariables().size());
            session.run(update);
            Operand result = instance.result();
            session.evaluate(new Integer[]{3,1}, tf.shape(result));
        }
    }
    
    @Test
    public void test_unweighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(new long[] {100, 40});
            MeanTensor instance = new MeanTensor(tf, TFloat64.DTYPE);
            session.run(tf.init());
            session.run(instance.initializeVars());
            Op update = instance.updateState(yPred);
            session.run(update);
            Operand result = instance.result();
            Float[] expected_result = new Float[] {100f, 40f};
            session.evaluate(expected_result, result);
            
            session.evaluate(expected_result, instance.getTotal());
            session.evaluate(new Float[]{1f,1f}, instance.getCount());
            
            session.run(instance.resetStates());
            session.evaluate(new Float[]{0f,0f}, instance.getTotal());
            session.evaluate(new Float[]{0f,0f}, instance.getCount());
            
        }
    }
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(new long[] {100, 30});
            MeanTensor instance = new MeanTensor(tf, TFloat64.DTYPE);
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            // check scalar weight
            Op update = instance.updateState(yPred, tf.constant(0.5f));
            session.run(update);
            Operand result = instance.result();
            Float[] expected_result = new Float[] {100f, 30f};
            session.evaluate(expected_result, result);
            session.evaluate(new Float[]{50f,15f}, instance.getTotal());
            session.evaluate(new Float[]{0.5f,0.5f}, instance.getCount());
            
            //check weights not scalar and weights rank matches values rank
            yPred = tf.constant(new long[] {1, 5});
            update = instance.updateState(yPred, tf.constant(new float[] {1f, 0.2f} ));
            session.run(update);
            result = instance.result();
            expected_result = new Float[] {51f / 1.5f, 16f / 0.7f};
            session.evaluate(expected_result, result);
            session.evaluate(new Float[]{51f,16f}, instance.getTotal());
            session.evaluate(new Float[]{1.5f,.7f}, instance.getCount());
            
            //check weights broadcast
            yPred = tf.constant(new long[] {1, 2});
            update = instance.updateState(yPred, tf.constant(0.5f));
            session.run(update);
            result = instance.result();
            expected_result = new Float[] {51.5f / 2f, 17f / 1.2f};
            session.evaluate(expected_result, result);
            session.evaluate(new Float[]{51.5f, 17f}, instance.getTotal());
            session.evaluate(new Float[]{2f, 1.2f}, instance.getCount());
            
            //check weights squeeze
            yPred = tf.constant(new long[] {1, 5});
            Operand sampleWeight = tf.constant(new float[][] {{1f}, {0.2f}});
            update = instance.updateState(yPred, sampleWeight);
            session.run(update);
            result = instance.result();
            expected_result = new Float[] {52.5f / 3f, 18f / 1.4f};
            session.evaluate(expected_result, result);
            session.evaluate(new Float[]{52.5f, 18f}, instance.getTotal());
            session.evaluate(new Float[]{3f, 1.4f}, instance.getCount());
            
           
        }
    }
    
     @Test
    public void test_weighted_expand() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
                        
            // check weights expand
            MeanTensor instance = new MeanTensor(tf, TFloat64.DTYPE);
            session.run(instance.initializeVars());
            Operand yPred = tf.constant(new long[][] {{1}, {5}});
            Operand sampleWeight = tf.constant(new float[] {1f, 0.2f});
            Op update = instance.updateState(yPred, sampleWeight);
            session.run(update);
            Operand result = instance.result();
            session.evaluate(tf.constant(new float[][] {{1f}, {5f}}), result);
            session.evaluate(tf.constant(new float[][]{{1f}, {1f}}), instance.getTotal());
            session.evaluate(tf.constant(new float[][]{{1f}, {0.2f}}), instance.getCount());
        }
    }
    
}
