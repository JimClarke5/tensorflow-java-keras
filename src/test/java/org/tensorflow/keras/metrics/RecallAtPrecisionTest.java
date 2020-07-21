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

import java.util.Random;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.random.RandomUniform;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author jbclarke
 */
public class RecallAtPrecisionTest {

    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;

    public RecallAtPrecisionTest() {
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
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            RecallAtPrecision instance = new RecallAtPrecision(tf, "recall_at_precision_1", 0.4f, 100);
            assertEquals("recall_at_precision_1", instance.getName());
            assertEquals(0.4f, instance.getPrecision());
            assertEquals(100, instance.getNumThresholds());
            assertEquals(4, instance.getVariables().size());
        }
    }

    @Test
    public void test_value_is_idempotent() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            RecallAtPrecision instance = new RecallAtPrecision(tf, 0.7f);
            session.run(instance.initializeVars());

            Operand yPred = tf.random.randomUniform(tf.constant(Shape.of(10, 3)), TFloat32.DTYPE, RandomUniform.seed(1L));
            Operand yTrue = tf.random.randomUniform(tf.constant(Shape.of(10, 3)), TFloat32.DTYPE, RandomUniform.seed(1L));
            yTrue = tf.math.mul(yTrue, tf.constant(2.0f));

            Op update = instance.updateState(yTrue, yPred);

            for (int i = 0; i < 10; i++) {
                session.run(update);
            }

            Operand initialPrecision = instance.result();

            for (int i = 0; i < 10; i++) {
                session.evaluate(initialPrecision, instance.result());
            }

        }
    }

    private Random random = new Random();

    private int[][] generateRandomArray(int dim1, int dim2, int maxVal) {
        int[][] result = new int[dim1][dim2];
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                result[i][j] = random.nextInt(maxVal);
            }
        }

        return result;
    }

    @Test
    public void test_unweighted_all_correct() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            RecallAtPrecision instance = new RecallAtPrecision(tf, 0.7f);
            session.run(instance.initializeVars());
            int[][] predArray = generateRandomArray(100, 1, 2);
            int[][] trueArray = new int[100][1]; // 100,1
            System.arraycopy(predArray, 0, trueArray, 0, predArray.length);
            Operand yPred = tf.constant(predArray);
            Operand yTrue = tf.constant(trueArray);
            yPred = tf.dtypes.cast(yPred, TFloat32.DTYPE);
            yTrue = tf.dtypes.cast(yTrue, TFloat32.DTYPE);

            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            Operand precision = instance.result();

            session.evaluate(1f, precision);
        }
    }

    @Test
    public void test_unweighted_high_precision() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            RecallAtPrecision instance = new RecallAtPrecision(tf, 0.75f);
            session.run(instance.initializeVars());
            Operand yPred = tf.constant(new float[]{
                0.05f, 0.1f, 0.2f, 0.3f, 0.3f, 0.35f, 0.4f, 0.45f, 0.5f, 0.6f, 0.9f, 0.95f});
            Operand yTrue = tf.constant(new long[]{0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1});

            Op update = instance.updateState(yTrue, yPred);
            session.run(update);

            Operand precision = instance.result();

            session.evaluate(0.5f, precision);
        }
    }

    @Test
    public void test_unweighted_low_precision() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            RecallAtPrecision instance = new RecallAtPrecision(tf, 2.0f / 3f);
            session.run(instance.initializeVars());
            Operand yPred = tf.constant(new float[]{
                0.05f, 0.1f, 0.2f, 0.3f, 0.3f, 0.35f, 0.4f, 0.45f, 0.5f, 0.6f, 0.9f, 0.95f});
            Operand yTrue = tf.constant(new long[]{0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1});

            Op update = instance.updateState(yTrue, yPred);
            session.run(update);

            Operand precision = instance.result();

            session.evaluate(5.f / 6f, precision);
        }
    }

    @Test
    public void test_weighted() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            RecallAtPrecision instance = new RecallAtPrecision(tf, 0.75f);
            session.run(instance.initializeVars());
            Operand yPred = tf.constant(new float[]{
                0.1f, 0.2f, 0.3f, 0.5f, 0.6f, 0.9f, 0.9f});
            Operand yTrue = tf.constant(new long[]{0, 1, 0, 0, 0, 1, 1});
            Operand sampleWeight = tf.constant(new float[]{1, 2, 1, 2, 1, 2, 1});

            Op update = instance.updateState(yTrue, yPred, sampleWeight);
            session.run(update);

            Operand precision = instance.result();

            session.evaluate(0.6f, precision);
        }
    }

    @Test
    public void test_unachievable_precision() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            RecallAtPrecision instance = new RecallAtPrecision(tf, 2.0f / 3f);
            session.run(instance.initializeVars());
            Operand yPred = tf.constant(new float[]{
                0.1f, 0.2f, 0.3f, 0.9f});
            Operand yTrue = tf.constant(new long[]{1, 1, 0, 0});

            Op update = instance.updateState(yTrue, yPred);
            session.run(update);

            Operand precision = instance.result();
            // The highest possible precision is 1/2 which is below the required
            // The highest possible precision is 1/2 which is below the required
            session.evaluate(0f, precision);
        }
    }

    @Test
    public void test_invalid_sensitivity() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            RecallAtPrecision instance = new RecallAtPrecision(tf, -1f);
            fail();
        } catch (AssertionError expected) {

        }
    }

    @Test
    public void test_invalid_num_thresholds() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            RecallAtPrecision instance = new RecallAtPrecision(tf, 0.7f, -1);
            fail();
        } catch (AssertionError expected) {

        }
    }
}
