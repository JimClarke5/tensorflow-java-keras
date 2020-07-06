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

/**
 *
 * @author jbclarke
 */
public class MeanRelativeErrorTest {
    
     private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
     float epsilon = 1e-4F;
    
    public MeanRelativeErrorTest() {
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
            
            MeanRelativeError instance = new MeanRelativeError(tf,"mre", new float[]{1,3});
            session.run(tf.init());
             instance.setDebug(null);
            assertEquals("mre", instance.getName());
            session.evaluate(tf.constant(new float[]{1,3}), instance.getNormalizer());
        }
    }

   @Test
    public void test_unweighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            float[][] predArray = new float[][] {{2, 4, 6, 8}};
            float[][] trueArray = new float[][] {{1, 3, 2, 3}};
            Operand yPred = tf.constant(predArray);
            Operand yTrue = tf.constant(trueArray);
            
            MeanRelativeError instance = new MeanRelativeError(tf, yTrue);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred );
            session.run(update);
            Operand result = instance.result();
            
            float expected_result = 1.25f;
            session.evaluate(expected_result, result);
         }
    }
    
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            float[] predArray = new float[] {2, 4, 6, 8};
            float[] trueArray = new float[] {1, 3, 2, 3};
            float[] sampleWeightArray = new float[] {0.2f, 0.3f, 0.5f, 0f};
            
            Operand yPred = tf.constant(predArray);
            Operand yTrue = tf.constant(trueArray);
            Operand sampleWeight = tf.constant(sampleWeightArray);
            
            MeanRelativeError instance = new MeanRelativeError(tf, yTrue);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred, sampleWeight );
            session.run(update);
            Operand result = instance.result();
            
            float expected_result = 1.3f;
            session.evaluate(expected_result, result);
         }
    }
    
    public void test_zero_normalizer() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            float[] predArray = new float[] {2, 4};
            int[] trueArray = new int[] {1, 3};
            
            Operand yPred = tf.constant(predArray);
            Operand yTrue = tf.constant(trueArray);
            
            MeanRelativeError instance = new MeanRelativeError(tf, tf.zerosLike(yTrue));
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred );
            session.run(update);
            Operand result = instance.result();
            
            float expected_result = 0f;
            session.evaluate(expected_result, result);
         }
    }
    
}
