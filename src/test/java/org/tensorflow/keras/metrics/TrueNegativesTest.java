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

/**
 *
 * @author jbclarke
 */
public class TrueNegativesTest {
    
       
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    float epsilon = 1e-4F;
    
    long[][] trueArray = {
                {0, 1, 0, 1, 0}, {0, 0, 1, 1, 1},
                {1, 1, 1, 1, 0}, {0, 0, 0, 0, 1}
            };
    
    long[][] predArray = {
                {0, 0, 1, 1, 0}, {1, 1, 1, 1, 1},
                {0, 1, 0, 1, 0}, {1, 1, 1, 1, 1}
            };
    
    float[] sampleWeightArray = {1.f, 1.5f, 2.f, 2.5f};
    
    public TrueNegativesTest() {
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
    public void test_config() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            TrueNegatives instance = new TrueNegatives(tf, "my_tn", new float[]{0.4f, 0.9f});
            assertEquals("my_tn", instance.getName());
            float[] expectedThresholds = new float[] {0.4f, 0.9f};
            assertArrayEquals(expectedThresholds, instance.getThresholds(),epsilon);
            assertEquals(1, instance.getVariables().size());
        }
    }
    
    @Test
    public void test_unweighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            
            Operand yPred = tf.constant(this.predArray);
            Operand yTrue = tf.constant(this.trueArray);
            TrueNegatives instance = new TrueNegatives(tf);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred );
            session.run(update);
            Operand result = instance.result();
            
            session.evaluate(3.0f, instance.result());
         }
    }
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            
            Operand yPred = tf.constant(this.predArray);
            Operand yTrue = tf.constant(this. trueArray);
            Operand sampleWeight = tf.constant(this.sampleWeightArray);
            TrueNegatives instance = new TrueNegatives(tf);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred, sampleWeight);
            session.run(update);
            Operand result = instance.result();
            
            session.evaluate(4.0f, instance.result());
         }
    }
    
    @Test
    public void test_unweighted_with_thresholds() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            
            Operand yPred = tf.constant(new float[][] {
                {0.9f, 0.2f, 0.8f, 0.1f},
                {0.2f, 0.9f, 0.7f, 0.6f},
                {0.1f, 0.2f, 0.4f, 0.3f},
                {0f, 1f, 0.7f, 0.3f}
            });
            Operand yTrue = tf.constant(new long[][] {
                {0, 1, 1, 0},
                {1, 0, 0, 0},
                {0, 0, 0, 0},
                {1, 1, 1, 1}
            });
            TrueNegatives instance = new TrueNegatives(tf, new float[]{0.15f, 0.5f, 0.85f});
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            Operand result = instance.result();
            Float[] expected = new Float[] {2.f, 5.f, 7.f};
            session.evaluate(expected, instance.result());
         }
    }
    
    @Test
    public void test_weighted_with_thresholds() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            
            Operand yPred = tf.constant(new float[][] {
                {0.9f, 0.2f, 0.8f, 0.1f},
                {0.2f, 0.9f, 0.7f, 0.6f},
                {0.1f, 0.2f, 0.4f, 0.3f},
                {0f, 1f, 0.7f, 0.3f}
            });
            Operand yTrue = tf.constant(new long[][] {
                {0, 1, 1, 0},
                {1, 0, 0, 0},
                {0, 0, 0, 0},
                {1, 1, 1, 1}
            });
            
            Operand sampleWeight = tf.constant(new float[][] {{  0.0f, 2.0f, 3.0f, 5.0f }});
            
            TrueNegatives instance = new TrueNegatives(tf, new float[]{0.15f, 0.5f, 0.85f});
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred, sampleWeight);
            session.run(update);
            Operand result = instance.result();
            Float[] expected = new Float[] {5.f, 15.f, 23.f};
            session.evaluate(expected, instance.result());
         }
    }
    
}
