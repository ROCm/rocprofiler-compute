.. meta::
   :description: ROCm Compute Profiler performance model
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, performance, model, profiler, tool, Instinct,
              accelerator, AMD

*****************
Performance model
*****************

ROCm Compute Profiler makes available an extensive list of metrics to better understand
achieved application performance on AMD Instinct™ MI-series accelerators
including Graphics Core Next™ (GCN) GPUs like the AMD Instinct MI50, CDNA™
accelerators like the MI100, CDNA2 accelerators such as the MI250X, MI250,
and MI210, and CDNA3 accelerators such as the MI300A, MI300X, MI325X, and MI350X. Refer to the table to get more details on the architectures: 

.. tab-set::

   .. tab-item:: CDNA

      .. raw:: html

         <div class="pst-scrollable-table-container">
            <table class="table">
               <thead>
               <tr class="row-odd">
                  <th class="head">
                     <p>Chip packaging</p>
                  </th>
                  <th class="head">
                     <p>Supported series</p>
                  </th>
                  <th class="head">
                     <p>Spatial partition modes</p>
                  </th>
                  <th class="head">
                     <p>Supported data operator type</p>
                  </th>
                  <th class="head">
                     <p>Data cache size</p>
                  </th>
                  <th class="head">
                     <p>Local data store size</p>
                  </th>
               </tr>
               </thead>
               <style>
               tbody#cdna-architecture-support tr:last-child {
                  border-bottom: 2px solid var(--pst-color-primary);
               }
               </style>
               <tbody id="cdna-architecture-support">
               <tr class="row-odd">
                  <td>
                     <p>Single Die</p>
                  </td>
                  <td>
                     <p>MI100</p>
                  </td>
                  <td>
                     <p>N/A</p>
                  </td>
                  <td>
                     <p>FP32</p>
                     <p>FP64</p>
                     <p>FP16</p>
                     <p>INT32 ADD/LOGIC/MAD</p>
                     <p>INT8 DOT</p>
                     <p>INT4 DOT</p>
                     <p>FP32 GEMM</p>
                  </td>
                  <td>
                     <p>L1 (KiB)</p>
                     <p>L2 Capacity (MiB)</p>
                     <p>L2 Cache line size (B)</p>
                     <p>Last level cache (MiB)</p>
                  </td>
                  <td>
                     <p>(KiB)</p>
                  </td>
               </tr>
               </tbody>
            </table>
         </div>
   
   .. tab-item:: CDNA 2

      .. raw:: html

         .. raw:: html

         <div class="pst-scrollable-table-container">
            <table class="table">
               <thead>
                     <tr>
                                 <th>Architecture</th>
                                 <th>CDNA</th>
                                 <th>CDNA2</th>
                                 <th>CDNA3</th>
                                 <th>Notes</th>
                             </tr>
                             <tr>
                                 <td>Chip Packaging</td>
                                 <td>Single Die</td>
                                 <td>2 Graphics Compute Dies (GCDs) into single package</td>
                                 <td>One logical processor with dozen chiplets, configurable with partition modes</td>
                                 <td rowspan="10">See top chip view in the below. See detailed node topology in each white paper.</td>
                             </tr>
                         </table>

                         <table border="1">
                             <tr>
                                 <th colspan="4" style="text-align:center;"><h3>Supported Series:</h3></th>
                             </tr>
                             <tr style="text-align:center;">
                                 <td>MI100</td>
                                 <td>MI200</td>
                                 <td>MI300A</td>
                                 <td>MI350X</td>
                             </tr>
                             <tr style="text-align:center;">
                                 <td>&nbsp;</td>
                                 <td>MI210</td>
                                 <td>MI300X</td>
                                 <td>&nbsp;</td>
                             </tr>
                             <tr style="text-align:center;">
                                 <td>&nbsp;</td>
                                 <td>MI250</td>
                                 <td>MI325X</td>
                                 <td>&nbsp;</td>
                             </tr>
                         </table>

                         <table border="1">
                             <tr>
                                 <th colspan="4" style="text-align:center;"><h3>Spatial Partition Modes:</h3></th>
                             </tr>
                         </table>

                         <table border="1">
                             <tr>
                                 <th colspan="4" style="text-align:center;"><h3>Supported Data Operator Type:</h3></th>
                             </tr>
                         </table>
                     </body>
                     </html>

   .. tab-item:: CDNA 3

      .. raw:: html

         .. raw:: html

         <div class="pst-scrollable-table-container">
            <table class="table">
               <thead>
               <tr class="row-odd">
                  <th class="head">
                     <p>Chip packaging</p>
                  </th>
                  <th class="head">
                     <p>Supported series</p>
                  </th>
                  <th class="head">
                     <p>Spatial partition modes</p>
                  </th>
                  <th class="head">
                     <p>Supported data operator type</p>
                  </th>
                  <th class="head">
                     <p>Data cache size</p>
                  </th>
                  <th class="head">
                     <p>Local data store size</p>
                  </th>
               </tr>
               </thead>
               <style>
               tbody#cdna-architecture-support tr:last-child {
                  border-bottom: 2px solid var(--pst-color-primary);
               }
               </style>
               <tbody id="cdna-architecture-support">
               <tr class="row-odd">
                  <td>
                     <p>Single Die</p>
                  </td>
                  <td>
                     <p>MI100</p>
                  </td>
                  <td>
                     <p>N/A</p>
                  </td>
                  <td>
                     <p>FP32</p>
                     <p>FP64</p>
                     <p>FP16</p>
                     <p>INT32 ADD/LOGIC/MAD</p>
                     <p>INT8 DOT</p>
                     <p>INT4 DOT</p>
                     <p>FP32 GEMM</p>
                  </td>
                  <td>
                     <p>L1 (KiB)</p>
                     <p>L2 Capacity (MiB)</p>
                     <p>L2 Cache line size (B)</p>
                     <p>Last level cache (MiB)</p>
                  </td>
                  <td>
                     <p>(KiB)</p>
                  </td>
               </tr>
               </tbody>
            </table>
         </div>

To best use profiling data, it's important to understand the role of various
hardware blocks of AMD Instinct accelerators. This section describes each
hardware block on the accelerator as interacted with by a software developer to
give a deeper understanding of the metrics reported by profiling data. Refer to
:doc:`/tutorial/profiling-by-example` for more practical examples and details on how
to use ROCm Compute Profiler to optimize your code.

.. _mixxx-note:

.. note::

   In this chapter, **MI2XX** refers to any of the CDNA2 architecture-based AMD
   Instinct MI250X, MI250, and MI210 accelerators interchangeably in cases
   where the exact product at hand is not relevant.

   For a comparison of AMD Instinct accelerator specifications, refer to
   :doc:`Hardware specifications <rocm:reference/gpu-arch-specs>`. For product
   details, see the :prod-page:`MI250X <mi200/mi250x>`,
   :prod-page:`MI250 <mi200/mi250>`, and :prod-page:`MI210 <mi200/mi210>`
   product pages.

In this chapter, the AMD Instinct performance model used by ROCm Compute Profiler is divided into a handful of
key hardware blocks, each detailed in the following sections:

* :doc:`compute-unit`

* :doc:`l2-cache`

* :doc:`shader-engine`

* :doc:`command-processor`

* :doc:`system-speed-of-light`
