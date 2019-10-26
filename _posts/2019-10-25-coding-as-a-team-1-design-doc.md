---
layout: post
title: "提升工程团队效率的习惯（一）：设计文档"
comments: true
---

在硅谷，街上随便叫住一个程序员，问：“在起一个project之前，你应该做什么？”估计十个程序员中有九个会回答：“写 design doc。”那么写 design doc 为什么这么重要呢？这还得从什么是 design doc 开始讲起。



设计文档（design doc）是一种工程师之间共享的文件，用来在写代码之前描述一个将要被实现的功能，比如这个[例子](https://github.com/sql-machine-learning/sqlflow/pull/1042)。设计文档往往由多个部分组成。

1. 概述。简单的几句话来描述将被实现的功能是什么？
2. 初衷。我们为什么要实现该功能？
3. 预览。该功能实现后会长什么样？
4. 技术选型。为了实现该功能，我们有哪些技术选择？每一种技术选择有什么优点和什么缺点？我们选择了其中哪一种，理由是什么？
5. 必要的实现细节。列举代码模版，如何测试，如何写日志等等。需要注意的是，设计文档应该保持 high level，具体实现是代码的工作。

一名主要作者会撰写这篇设计文档，作者的同僚们也会通过在这份文档中评论的方式和作者进行交流。整个交流过程往往持续几天甚至好几周。当作者与同僚们就文档所涉及的内容达成一致后，这份文档就可以被合入代码库了。这也标志着设计文档所描述的项目的开始。



在项目中加入设计文档的这一套流程有许许多多的好处。

第一，设计文档能帮你理清思路。如果你能把你的想法用简练的逻辑清晰的文字表述出来，这说明你对整个功能的实现是有深刻的理解的。反之，如果你连自己想写什么都表达不清楚，那就更别提实现功能了。你的同僚们在给你设计文档评论的过程中，也会帮助你一起检查你的思路，提前发现问题，指出你逻辑的漏洞，甚至提出更好的实现方法（我一次又一次地发现周围的同事总能提出更好地解决方案，指出我逻辑的漏洞）。尤其是团队中那些经验丰富的程序员能通过简单的建议帮你省下日后大量的开发时间。

第二，设计文档能帮助整个团队达成一致。它能让每个人的技术思路变得透明，代码库也会变得更加可维护。如果设计文档是大家一致通过的，那大家对整个项目就会更有 ownership，也会更乐意帮助提升代码质量，整个项目的工程质量就会很高。

第三，设计文档是对代码的一个很好的补充。团队新人和别组同事很容易通过设计文档对整个代码有 high level 的理解。当别人想向你了解项目情况的时候，再也没有比甩过去一个链接更让人痛快的了。

第四，设计文档能够帮助团队节省时间。比起冗长的会议，设计文档更能让（不同时区的）同僚们能按照自己适宜的时间，静下心来独立思考，并把自己的想法提炼出来反馈给作者。开发前，同僚之间的讨论，能够很好地检查整个设计的思路，大大降低错误设计的风险（相信我，我曾经不止一次偷懒了直接写 code，导致自己好几天的开发到最后被证明是无用的）。开发中，由于同僚之间都已经了解设计思路，code review 也会变得简单。如果预开发中遇到了问题，开发者也更容易得到同僚的帮助。开发后，设计文档也为整个团队做技术复盘提供了依据。

第五，设计文档能很好的组织团队。参与讨论的同僚们往往是对设计文档所要解决的问题感兴趣的人。通过设计文档，他们能快速组建小团队，并以足够的激情高效地完成任务。

第六，设计文档是很好的晋升素材。设计文档往往体现出其作者的工程素养，思考问题的清晰程度，以及解决重要问题的能力。



既然写设计文档能给团队带来这么多的好处，为什么不是所有人都跟着这样做呢？据我观察，可能是因为如下几种认知：

1. “写设计文档太麻烦了，要不先 coding 再说？”如果觉得设计文档很难写，就说明作者对问题理解还不够透彻，那么其所选择的实现方法很可能是错误的。刚开始写设计文档的确会花费大量的精力，但就如上文所述，从长期来看，这些投入是有价值的。
2. “省去写文档的功夫，我代码早就写完了。”凡事都要有度，是否需要设计文档取决于项目长期的代码质量和可维护性。如果功能很容易实现，且其实现逻辑很直接，那也许建立一个 issue 简单地描述功能即可，代码的维护和团队思路的一致性可以通过 code review 来实现。但是，如果功能很复杂，还是推荐先写设计文档。
3. “这活儿我擅长，看我直接把它实现了，可以 impress 同事们。”有自己擅长的领域固然加分，但团队内的沟通也同样重要。一方面，如果同僚们不了解作者的思路，他们很难做 code review；另一方面，代码的维护者和作者往往不是同一个人，如果作者的思路没有通过文档保留下来，代码又比较复杂，整个项目的可维护性就会下降。



努力去改变这些认知，踏出舒适圈，尝试开始写设计文档，积极参与评论别人的设计文档，你会发现，你在团队中的贡献会成倍增加。



一些相关的事情：

1. [@reyoung](https://github.com/reyoung) 老师的博文：[《文档是沟通的中间件》](https://github.com/tonyyang-svail/blog/blob/tonyyang-svail-fix-why-documentation-links/2016/11/why_documentation.md)
2. [sql-machine-learning 项目](https://github.com/sql-machine-learning)中的一些例子
   1. [Customized Model](https://github.com/sql-machine-learning/sqlflow/blob/develop/doc/design/design_customized_model.md)
   2. [Parameter Server](https://github.com/sql-machine-learning/elasticdl/blob/c6b4ee5ec0a692b8142426df4c9c104a24864194/docs/designs/ps_design.md)
3. 我曾向同事吐槽说：“我真是看不懂人家写的代码，尤其是 Python 这种容易写不容易读的语言写的代码。”同事回：“放心，你写的他们也看不懂”。我当时就乐了，想：如果没有设计文档让大家在 high level 保持一致的话，每个人只是读对方的代码进行交流的话，团队的运转会是多么的低效呀。