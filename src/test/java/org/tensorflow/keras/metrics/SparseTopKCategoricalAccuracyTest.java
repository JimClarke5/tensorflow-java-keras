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
import org.tensorflow.types.TInt32;

/**
 *
 * @author jbclarke
 */
public class SparseTopKCategoricalAccuracyTest {
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public SparseTopKCategoricalAccuracyTest() {
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
            SparseTopKCategoricalAccuracy instance = new SparseTopKCategoricalAccuracy(tf, "stopkca", TInt32.DTYPE);
            assertEquals("stopkca", instance.getName());
            assertEquals(TInt32.DTYPE, instance.getDataType());
        }
    }
    
    @Test
    public void test_correctness() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            SparseTopKCategoricalAccuracy instance = new SparseTopKCategoricalAccuracy(tf);
            session.run(instance.initializeVars());
            
            Operand yTrue = tf.constant(new float[] {2,1});
            Operand yPred = tf.constant(new float[][] {{0.1f, 0.9f, 0.8f}, {0.05f, 0.95f, 0f}});
            
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            session.evaluate(1.f, instance.result());
            
            //With `k` < 5.
            instance = new SparseTopKCategoricalAccuracy(tf, 1);
            session.run(instance.initializeVars());
            update = instance.updateState(yTrue, yPred);
            session.run(update);
            session.evaluate(0.5f, instance.result());
            
            //With `k` > 5.
            yPred = tf.constant(new float[][] {
                {0.5f, 0.9f, 0.1f, 0.7f, 0.6f, 0.5f, 0.4f}, 
                {0.05f, 0.95f, 0f, 0f, 0f, 0f, 0f}});
            instance = new SparseTopKCategoricalAccuracy(tf, 6);
            session.run(instance.initializeVars());
            update = instance.updateState(yTrue, yPred);
            session.run(update);
            session.evaluate(0.5f, instance.result());
            
        }
    }
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            SparseTopKCategoricalAccuracy instance = new SparseTopKCategoricalAccuracy(tf,2);
            session.run(instance.initializeVars());
            
            Operand yTrue = tf.constant(new float[] {1, 0, 2});
            Operand yPred = tf.constant(new float[][] {
                {0f, 0.9f, 0.1f}, 
                {0f, 0.9f, 0.1f},
                {0f, 0.9f, 0.1f}
            });
            
            Operand sampleWeight = tf.constant(new float[] {1, 0, 1});
            
            Op update = instance.updateState(yTrue, yPred, sampleWeight);
            session.run(update);
            session.evaluate(1.f, instance.result());
        }
    }
}
