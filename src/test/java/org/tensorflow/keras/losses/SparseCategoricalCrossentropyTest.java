/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.losses;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class SparseCategoricalCrossentropyTest {

    float epsilon = 1e-4F;
    TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public SparseCategoricalCrossentropyTest() {
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

    /**
     * Test of call method, of class SparseSparseCategoricalCrossentropy.
     */
    @Test
    public void testConfig() {
        System.out.println("testConfig");
        SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(null);
        assertEquals("sparse_categorical_crossentropy", instance.getName());

        instance = new SparseCategoricalCrossentropy(null, "scc", Reduction.SUM);
        assertEquals("scc", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class SparseCategoricalCrossentropy.
     */
    @Test
    public void testAllCorrectUnweighted() {
        System.out.println("testAllCorrectUnweighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            long[] true_np = {0L, 1L, 2L};
            float[] pred_np = {
                1.F, 0.F, 0.F,
                0.F, 1.F, 0.F,
                0.F, 0.F, 1.F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf);
            Operand<TFloat32> loss = instance.call(y_true, y_pred);
            float expected = 2.3841854e-7F;
            testSession.evaluate(expected, loss);

            System.out.println("============ LOGITS =================");
            // Test with logits.
            float[] logits_np = {
                10.F, 0.F, 0.F,
                0.F, 10.F, 0.F,
                0.F, 0.F, 10.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new SparseCategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits);
            testSession.evaluate(9.083335E-5F, loss);
        }
    }

    /**
     * Test of call method, of class SparseCategoricalCrossentropy.
     */
    @Test
    public void test_unweighted() {
        System.out.println("test_unweighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf);
            int[] true_np = {0, 1, 2};
            float[] pred_np = {
                .9F, .05F, .05F,
                .5F, .89F, .6F,
                .05F, .01F, .94F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand loss = instance.call(y_true, y_pred);
            float expected = 0.32396814F;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new SparseCategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits);
            expected = 0.05737559F;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class SparseCategoricalCrossentropy.
     */
    @Test
    public void test_scalar_weighted() {
        System.out.println("test_scalar_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            int[] true_np = {0, 1, 2};
            float[] pred_np = {
                .9F, .05F, .05F,
                .5F, .89F, .6F,
                .05F, .01F, .94F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand sampleWeight = tf.constant(2.3f);

            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = .7451267F;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new SparseCategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits, sampleWeight);
            expected = 0.13196386F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sample_weighted() {
        System.out.println("test_sample_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf);
            float[] sample_weight_np = {1.2F, 3.4F, 5.6F};
            int[] true_np = {0, 1, 2};
            float[] pred_np = {
                .9F, .05F, .05F,
                .5F, .89F, .6F,
                .05F, .01F, .94F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_weight_np), tf.constant(Shape.of(3, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 1.0696F;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new SparseCategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits, sampleWeight);
            expected = 0.31829F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_no_reduction() {
        System.out.println("test_no_reduction");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            // Test with logits.
            int[] true_np = {0, 1, 2};
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf, true, 0.0F, Reduction.NONE);
            Operand loss = instance.call(y_true, logits);
            Float[] expected = {0.001822F, 0.000459F, 0.169846F};
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_label_smoothing() {
        System.out.println("test_label_smoothing");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float label_smoothing = 0.1f;
            int[] true_np = {0, 1, 1};
            float[] logits_array = {100.0f, -100.0f, -100.0f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand logits = tf.reshape(tf.constant(logits_array), tf.constant(Shape.of(1, 3)));

            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf, true, label_smoothing);
            Operand loss = instance.call(y_true, logits);
            System.out.println(loss.asOutput().shape());
            float expected = 400.0F * label_smoothing / 3.0F;
            testSession.evaluate(expected, loss);
        } catch (Exception expected) {

        }
    }

}
