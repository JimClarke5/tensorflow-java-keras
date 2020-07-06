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
import org.tensorflow.op.core.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;

/**
 *
 * @author jbclarke
 */
public class MeanIoUTest {
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    float epsilon = 1e-4F;
    private long numClasses = 2l;
    
    public MeanIoUTest() {
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
            
            MeanIoU instance = new MeanIoU(tf,"iou", numClasses);
            session.run(tf.init());
            assertEquals("iou", instance.getName());
            assertEquals(numClasses, instance.getNumClasses());
            assertEquals(1, instance.getVariables().size());
        }
    }
    
    @Test
    public void test_unweighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(new long[] {0, 1, 0, 1});
            Operand yTrue = tf.constant(new long[] {0, 0, 1, 1});
            MeanIoU instance = new MeanIoU(tf, numClasses);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred );
            session.run(update);
            Operand result = instance.result();
            float expected_result = (1f / (2f + 2f - 1f) + 1f / (2f + 2f - 1f)) / 2f;
            session.evaluate(expected_result, result);
        }
    }
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(new long[] {0, 1, 0, 1});
            Operand yTrue = tf.constant(new long[] {0, 0, 1, 1});
            Operand sampleWeight = tf.constant(new float[] {0.2f, 0.3f, 0.4f, 0.1f});
            MeanIoU instance = new MeanIoU(tf, numClasses);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred, sampleWeight );
            session.run(update);
            Operand result = instance.result();
            float expected_result = (0.2f / (0.6f + 0.5f - 0.2f) + 0.1f / (0.4f + 0.5f - 0.1f)) / 2f;
            session.evaluate(expected_result, result);
        }
    }
    
    @Test
    public void test_multi_dim_input() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(new long[][] {{0, 1}, {0, 1}});
            Operand yTrue = tf.constant(new long[][] {{0, 0}, {1, 1}});
            Operand sampleWeight = tf.constant(new float[][] {{0.2f, 0.3f}, {0.4f, 0.1f}});
            MeanIoU instance = new MeanIoU(tf, numClasses);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred, sampleWeight );
            session.run(update);
            Operand result = instance.result();
            float expected_result = (0.2f / (0.6f + 0.5f - 0.2f) + 0.1f / (0.4f + 0.5f - 0.1f)) / 2f;
            session.evaluate(expected_result, result);
        }
    }
    
     @Test
    public void  test_zero_valid_entries() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            MeanIoU instance = new MeanIoU(tf, numClasses);
            session.run(tf.init());
            Operand result = instance.result();
            session.evaluate(0.0f, result); 
        }
    }
    
    @Test
    public void test_zero_and_non_zero_entries() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(new float[] {1});
            Operand yTrue = tf.constant(new int[] {1});
            
            MeanIoU instance = new MeanIoU(tf, numClasses);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred );
            session.run(update);
            Operand result = instance.result();
            float expected_result = (0f + 1f / (1f + 1f - 1f)) / 1f;
            session.evaluate(expected_result, result);
        }
    }

    
}
