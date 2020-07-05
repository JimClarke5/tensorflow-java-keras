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
package org.tensorflow.keras.activations;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.PrintUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;

/**
 *
 * @author Jim Clarke
 */
public class SwishTest {

    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;

    public SwishTest() {
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
     * Test of Swish call method
     */
    @Test
    public void testCall_Int() {
        int[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        int[] actual = {0, 0, 0, 0, 0, 0, 0, 0};
        int[] expected = {};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Swish<TInt32> instance = new Swish<>(tf);
            Operand<TInt32> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTInt32(operand.asTensor());
            assertArrayEquals(expected, actual);
            fail();
        } catch (AssertionError ex) {
            // expected
            //fail(ex);
        }
    }

    /**
     * Test of Swish call method
     */
    @Test
    public void testCall__Float() {
        float[] input = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] actual = new float[input.length];
        float[] expected = {
            0.7310586F, 1.7615942F, 2.8577223F, 3.928055F, 4.9665356F, 5.985164F, 6.993623F, 7.9973164F};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Swish<TFloat32> instance = new Swish<>(tf);
            Operand<TFloat32> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }

    /**
     * Test of Swish call method
     */
    @Test
    public void testCall__Double() {
        double[] input = {1, 2, 3, 4, 5, 6, 7, 8};
        double[] actual = {0, 0, 0, 0, 0, 0, 0, 0};
        double[] expected = {
            0.7310585786300049, 1.7615941559557646, 2.8577223804673,
            3.928055160151634, 4.966535745378576, 5.985164261060192,
            6.993622641639195, 7.997317198956269};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Swish<TFloat64> instance = new Swish<>(tf);
            Operand<TFloat64> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON);
        }
    }

}
