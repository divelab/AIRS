<p align="center">
<img src="https://github.com/divelab/AIRS/blob/main/AIRS_logo.png" width="225" class="center" alt="logo"/>
    <br/>
</p>

[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg
[license-url]:https://github.com/divelab/AIRS/blob/main/LICENSE
[contributing-image]:https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat


![Last Commit](https://img.shields.io/github/last-commit/divelab/AIRS)
[![License][license-image]][license-url]
![Contributing][contributing-image]

------

# Artificial Intelligence for Science (AIRS)

AIRS is a collection of open-source software tools, datasets, and benchmarks associated with our paper entitled “Artificial Intelligence for Science in Quantum, Atomistic, and Continuum Systems”. Our focus here is on AI for Science, including AI for quantum, atomistic, and continuum systems. Our goal is to develop and maintain an integrated, open, reproducible, and sustainable set of resources in order to propel the emerging and rapidly growing field of AI for Science. The list of resources will grow as our research progresses. The current list includes:
-	[**OpenQM**](https://github.com/divelab/AIRS/tree/main/OpenQM): AI for Quantum Mechanics
-	[**OpenDFT**](https://github.com/divelab/AIRS/tree/main/OpenDFT): AI for Density Functional Theory
-	[**OpenMol**](https://github.com/divelab/AIRS/tree/main/OpenMol): AI for Small Molecules
-	[**OpenProt**](https://github.com/divelab/AIRS/tree/main/OpenProt): AI for Protein Science
-	[**OpenMat**](https://github.com/divelab/AIRS/tree/main/OpenMat): AI for Materials Science
-	[**OpenMI**](https://github.com/divelab/AIRS/tree/main/OpenMI): AI for Molecular Interactions
-	[**OpenPDE**](https://github.com/divelab/AIRS/tree/main/OpenPDE): AI for Partial Differential Equations



<p align="center">
<img src="https://github.com/divelab/AIRS/blob/main/overview.jpeg" width="950" class="center" alt="logo"/>
    <br/>
</p>

## Methods
Here is the summary of methods we have in AIRS. More methods will be included as our research progresses.

<table>
  <thead>
    <tr>
      <th> Quantum</th>
      <th colspan="2"> Atomistic</th>
      <th> Continuum</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td> 
          <img src="https://placehold.co/50x50/87ffac/87ffac.png" height="12" width="12"> <b>OpenQM</b>
          <ul>
            <li><a href="OpenQM/LCN">LCN</a></li>
            <li><a href="OpenQM/DiffVMC">DiffVMC</a></li>
          </ul>
          <img src="https://placehold.co/50x50/43E976/43E976.png" height="12" width="12"> <b>OpenDFT</b>
          <ul>
            <li><a href="OpenDFT/QHNet">QHNet</a></li>
            <li><a href="OpenDFT/QHBench">QHBench</a></li>
          </ul>
      </td>
      <td> 
          <img src="https://placehold.co/50x50/EEC0FF/EEC0FF.png" height="12" width="12"> <b>OpenMol</b>
          <ul>
            <li><a href="OpenMol/SphereNet">SphereNet</a></li>
            <li><a href="OpenMol/ComENet">ComENet</a></li>
            <li><a href="OpenMol/GraphDF">GraphDF</a></li>
            <li><a href="OpenMol/G-SphereNet">G-SphereNet</a></li>
          </ul>
          <img src="https://placehold.co/50x50/D790FF/D790FF.png" height="12" width="12"> <b>OpenProt</b>
          <ul>
            <li><a href="OpenProt/ProNet">ProNet</a></li>
          </ul>
      </td>
      <td>
          <img src="https://placehold.co/50x50/B174E9/B174E9.png" height="12" width="12"> <b>OpenMat</b>
          <ul>
            <li><a href="OpenMat/Matformer">Matformer</a></li>
            <li><a href="OpenMat/PotNet">PotNet</a></li>
            <li><a href="OpenMat/SyMat">SyMat</a></li>
          </ul>
          <img src="https://placehold.co/50x50/8D55F7/8D55F7.png" height="12" width="12"> <b>OpenMI</b>
          <ul>
            <li><a href="OpenMI/GraphBP">GraphBP</a></li>
          </ul>
      </td>
      <td> 
          <img src="https://placehold.co/50x50/FFA76E/FFA76E.png" height="12" width="12"> <b>OpenPDE</b>
          <ul>
            <li><a href="OpenPDE/G-FNO">G-FNO</a></li>
          </ul>
      </td>
    </tr>
  </tbody>
</table>


## Survey paper

**[Paper](https://arxiv.org/abs/2307.08423)** | **[Website](https://www.air4.science/)**

We released our survey paper about artificial intelligence for science in quantum, atomistic, and continuum systems. Please check our paper and website for more details!

### Citation

To cite the survey paper, please use the BibTeX entry provided below. There are two versions: the short version with part of the author names and the other with all author names.

```
@article{zhang2023artificial,
  title={Artificial Intelligence for Science in Quantum, Atomistic, and Continuum Systems},
  author={Xuan Zhang and Limei Wang and Jacob Helwig and Youzhi Luo and Cong Fu and Yaochen Xie and {...} and Shuiwang Ji},
  journal={arXiv preprint arXiv:2307.08423},
  year={2023}
}

@article{zhang2023artificial,
  title={Artificial Intelligence for Science in Quantum, Atomistic, and Continuum Systems},
  author={Xuan Zhang and Limei Wang and Jacob Helwig and Youzhi Luo and Cong Fu and Yaochen Xie and Meng Liu and Yuchao Lin and Zhao Xu and Keqiang Yan and Keir Adams and Maurice Weiler and Xiner Li and Tianfan Fu and Yucheng Wang and Haiyang Yu and YuQing Xie and Xiang Fu and Alex Strasser and Shenglong Xu and Yi Liu and Yuanqi Du and Alexandra Saxton and Hongyi Ling and Hannah Lawrence and Hannes St{\"a}rk and Shurui Gui and Carl Edwards and Nicholas Gao and Adriana Ladera and Tailin Wu and Elyssa F. Hofgard and Aria Mansouri Tehrani and Rui Wang and Ameya Daigavane and Montgomery Bohde and Jerry Kurtin and Qian Huang and Tuong Phung and Minkai Xu and Chaitanya K. Joshi and Simon V. Mathis and Kamyar Azizzadenesheli and Ada Fang and Al{\'a}n Aspuru-Guzik and Erik Bekkers and Michael Bronstein and Marinka Zitnik and Anima Anandkumar and Stefano Ermon and Pietro Li{\`o} and Rose Yu and Stephan G{\"u}nnemann and Jure Leskovec and Heng Ji and Jimeng Sun and Regina Barzilay and Tommi Jaakkola and Connor W. Coley and Xiaoning Qian and Xiaofeng Qian and Tess Smidt and Shuiwang Ji},
  journal={arXiv preprint arXiv:2307.08423},
  year={2023}
}
```
