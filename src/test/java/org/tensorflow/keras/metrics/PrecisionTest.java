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
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

/**
 *
 * @author jbclarke
 */
public class PrecisionTest {
     private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    float epsilon = 1e-4F;
    
    public PrecisionTest() {
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
            
            Precision instance = new Precision(tf,"my_precision", 
                    new float[] {0.4f, 0.9f}, 15,12);
            session.run(tf.init());
            session.run(instance.initializeVars());
            assertEquals("my_precision", instance.getName());
            assertEquals(2, instance.getVariables().size());
            assertArrayEquals(new float[] {0.4f, 0.9f}, instance.getThresholds());
            assertEquals(15, instance.getTopK());
            assertEquals(12, instance.getClassId());
        }
    }
    
    @Test
    public void test_value_is_idempotent() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            
            Precision instance = new Precision(tf, new float[] {0.3f, 0.72f});
            session.run(tf.init());
            session.run(instance.initializeVars());
            Operand yPred = tf.random.randomUniform(tf.constant(Shape.of(10,3)), TFloat32.DTYPE);
            Operand yTrue = tf.random.randomUniform(tf.constant(Shape.of(10,3)), TFloat32.DTYPE);
            
            Op update = instance.updateState(yTrue, yPred );
            
             for(int i = 0; i < 10; i++) {
                session.run(update);
            }
            
            Operand initialPrecision  = instance.result();
            
             for(int i = 0; i < 10; i++) {
                 session.evaluate(initialPrecision, instance.result());
            }
        }
    }
    
    @Test
    public void trst_unweighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Precision instance = new Precision(tf);
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new long[][] {{1, 0, 1, 0}});
            Operand yTrue = tf.constant(new long[][] {{0, 1, 1, 0}});
            Op update = instance.updateState(yTrue, yPred );
            session.run(update);
            Operand precision  = instance.result();
            session.evaluate(5.0f, precision);
            
        }
    }
    
    @Test
    public void trst_unweighted_all_incorrect() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Precision instance = new Precision(tf, new float[]{0.5f} );
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.random.randomUniform(tf.constant(Shape.of(100, 1)), TInt32.DTYPE);
            Operand yTrue = tf.math.sub(tf.constant(1), yPred);
            Op update = instance.updateState(yTrue, yPred );
            session.run(update);
            Operand precision  = instance.result();
            session.evaluate(0.0f, precision);
            
        }
    }
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Precision instance = new Precision(tf);
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new long[][] {{1, 0, 1, 0}, {1, 0, 1, 0} });
            Operand yTrue = tf.constant(new long[][] {{0, 1, 1, 0}, {1, 0, 0, 1} });
            Operand sampleWeight = tf.constant(new long[][] {{1, 2, 3, 4}, {4, 3, 2, 1} });
            Op update = instance.updateState(yTrue, yPred, sampleWeight );
            session.run(update);
            Operand precision  = instance.result();
            
            float weighted_tp = 3.0f + 4.0f;
            float weighted_positives = (1.0f + 3.0f) + (4.0f + 2.0f);
            float expected_precision = weighted_tp / weighted_positives;
            
            session.evaluate(expected_precision, precision);
            
        }
    }
    
    @Test
    public void test_div_by_zero() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Precision instance = new Precision(tf);
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new int[] {0,0,0,0});
            Operand yTrue = tf.constant(new int[] {0,0,0,0});
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            Operand precision  = instance.result();
            
            session.evaluate(0f, precision);
        }
    }
    
    @Test
    public void test_unweighted_with_threshold() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Precision instance = new Precision(tf, new float[] {0.5f, 0.7f});
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[][] {{1f, 0f, 0.6f, 0f}});
            Operand yTrue = tf.constant(new long[][] {{0, 1, 1, 0}});
            Operand sampleWeight = tf.constant(new long[][] {{1, 2, 3, 4}});
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            Operand precision  = instance.result();
            
            Float[] expected = new Float[] {0.5f, 0.f };
            
            session.evaluate(expected, precision);
            
        }
    }
    
    @Test
    public void test_weighted_with_threshold() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Precision instance = new Precision(tf, new float[] {0.5f, 1.f});
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[][] {{1f, 0f}, {0.6f, 0f}});
            Operand yTrue = tf.constant(new long[][] {{0, 1}, {1, 0}});
            Operand sampleWeight = tf.constant(new float[][] {{4, 0}, {3, 1}});
            Op update = instance.updateState(yTrue, yPred, sampleWeight );
            session.run(update);
            Operand precision  = instance.result();
            
            float weighted_tp = 0f + 3.f;
            float weighted_positives = (0f + 3.f) + (4.f + 0.f);
            float expected_precision = weighted_tp / weighted_positives;
            
            Float[] expected =  new Float[]{expected_precision, 0f};
            session.evaluate(expected, precision);
            
        }
    }
    
    @Test
    public void test_multiple_updates() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Precision instance = new Precision(tf, new float[] {0.5f, 1.f});
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[][] {{1f, 0f}, {0.6f, 0f}});
            Operand yTrue = tf.constant(new long[][] {{0, 1}, {1, 0}});
            Operand sampleWeight = tf.constant(new float[][] {{4, 0}, {3, 1}});
            Op update = instance.updateState(yTrue, yPred, sampleWeight );
            for(int i = 0 ; i < 2; i++)
                session.run(update);
            Operand precision  = instance.result();
            
            float weighted_tp = (0f + 3.f) + (0f + 3.f);
            float weighted_positives = ((0f + 3.f) + (4.f + 0.f)) + ((0f + 3.f) + (4.f + 0.f));
            float expected_precision = weighted_tp / weighted_positives;
            
            Float[] expected =  new Float[]{expected_precision, 0f};
            session.evaluate(expected, precision);
            
        }
    }
    
    @Test
    public void test_unweighted_top_k() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            // set topK to 3
            Precision instance = new Precision(tf, null, 3, null);
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[][] {{0.2f, 0.1f, 0.5f, 0f, 0.2f}});
            Operand yTrue = tf.constant(new long[][] {{0, 1, 1, 0, 0}});
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            Operand precision  = instance.result();
            session.evaluate(1.0f/ 3.0f, precision);
        }
    }
    
    @Test
    public void test_weighted_top_k() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            // set topK to 3
            Precision instance = new Precision(tf, null, 3, null);
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[] {0.2f, 0.1f, 0.4f, 0f, 0.2f});
            Operand yTrue = tf.constant(new long[] {0, 1, 1, 0, 1});
            Operand sampleWeight = tf.constant(new long[][] {{1, 4, 2, 3, 5}});
            Op update = instance.updateState(yTrue, yPred, sampleWeight);
            session.run(update);
            
            yPred = tf.constant(new float[][] {{0.2f, 0.6f, 0.4f, 0.2f, 0.2f}});
            yTrue = tf.constant(new long[][] {{1, 0, 1, 1, 1}});
            update = instance.updateState(yTrue, yPred, tf.constant(3));
            session.run(update);
            
            Operand precision  = instance.result();
            
            float tp = (2f + 5f) + (3f + 3f);
            float predicted_positives = (1f + 2f + 5f) + (3f + 3f + 3f);
            float expected_precision = tp / predicted_positives;
            session.evaluate(expected_precision, precision);
        }
    }
    
    @Test
    public void test_unweighted_class_id() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            // set classId to 2
            Precision instance = new Precision(tf, null, null, 2);
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[][] {{0.2f, 0.1f, 0.6f, 0f, 0.2f}});
            Operand yTrue = tf.constant(new long[][] {{0, 1, 1, 0, 0}});
            //instance.setDebug(session.getGraphSession());
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            Operand precision = instance.result();
            
            session.evaluate(1, precision);
            session.evaluate(1, instance.getTruePositives());
            session.evaluate(0, instance.getFalsePositives());
            //instance.setDebug(null);
            
            yPred = tf.constant(new float[][] {{0.2f, 0.1f, 0f, 0f, 0.2f}});
            yTrue = tf.constant(new long[][] {{0, 1, 1, 0, 0}});
            update = instance.updateState(yTrue, yPred);
            session.run(update);
            precision = instance.result();
            
            session.evaluate(1, precision);
            session.evaluate(1, instance.getTruePositives());
            session.evaluate(0, instance.getFalsePositives());
            
            yPred = tf.constant(new float[][] {{0.2f, 0.1f, 0.6f, 0f, 0.2f}});
            yTrue = tf.constant(new long[][] {{0, 1, 0, 0, 0}});
            update = instance.updateState(yTrue, yPred);
            session.run(update);
            precision = instance.result();
            
            session.evaluate(0.5, precision);
            session.evaluate(1, instance.getTruePositives());
            session.evaluate(1, instance.getFalsePositives());
        }
    }
    
    @Test
    public void test_unweighted_top_k_and_class_id() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            // set classId to 2
            Precision instance = new Precision(tf, null, 2, 2);
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[][] {{0.2f, 0.6f, 0.3f, 0f, 0.2f}});
            Operand yTrue = tf.constant(new long[][] {{0, 1, 1, 0, 0}});
            //instance.setDebug(session.getGraphSession());
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            Operand precision = instance.result();
            
            session.evaluate(1, precision);
            session.evaluate(1, instance.getTruePositives());
            session.evaluate(0, instance.getFalsePositives());
            //instance.setDebug(null);
            
            yPred = tf.constant(new float[][] {{1f, 1f, 0.9f, 1f, 1f}});
            yTrue = tf.constant(new long[][] {{0, 1, 1, 0, 0}});
            update = instance.updateState(yTrue, yPred);
            session.run(update);
            precision = instance.result();
            
            session.evaluate(1, precision);
            session.evaluate(1, instance.getTruePositives());
            session.evaluate(0, instance.getFalsePositives());
            
        }
    }
    @Test
    public void test_unweighted_top_k_and_threshold() {
       try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            // set topK to 2
            Precision instance = new Precision(tf, 0.7f, 2, null);
            session.run(tf.init());
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[][] {{0.2f, 0.8f, 0.6f, 0f, 0.2f}});
            Operand yTrue = tf.constant(new long[][] {{0, 1, 1, 0, 1}});
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            Operand precision = instance.result();
            
            session.evaluate(1, precision);
            session.evaluate(1, instance.getTruePositives());
            session.evaluate(0, instance.getFalsePositives());
            
            
        }
    }
    
    
    
}
