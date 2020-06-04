---
layout: post
title: Kaggle Competition Writeup - IEEE CIS Fraud Detection
description: "This is a write-up of a presentation on generating music in the waveform domain, which was part of a tutorial that I co-presented at ISMIR 2019 earlier this month."

tags: [machine learning, kaggle competition]

image:
  feature:
comments: false
share: true
---

During this period of lockdown I decided to attempt the IEEE CIS Fraud Detection Kaggle Competition. The ideas that I used for this competition were very well received, so I've decided to write it up in the form of a blog post.

## <a name="overview"></a> Overview

This blog post is divided into a few different sections. I'll try to motivate why modelling music in the waveform domain is an interesting problem. Then I'll give an overview of generative models, the various flavours that exist, and some important ways in which they differ from each other. In the next two sections I'll attempt to cover the state of the art in both likelihood-based and adversarial models of raw music audio. Finally, I'll raise some observations and discussion points. If you want to skip ahead, just click the section title below to go there.

* *[Problem Statement](#motivation)*
* *[Dataset](#dataset)*
* *[Exploration](#exploration)*
* *[Feature Engineering](#feature-engineering)*
* *[Discussion](#discussion)*
* *[Conclusion](#conclusion)*
* *[References](#references)*

Note that this blog post is not intended to provide an exhaustive overview of all the published research in this domain -- I have tried to make a selection and I've inevitably left out some great work. **Please don't hesitate to suggest relevant work in the comments section!**


## <a name="motivation"></a> Problem Statement

In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud.

The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.

## <a name="dataset"></a> Dataset

The description of the dataset provided in the competitions page was very brief. Below is the more detailed version of the dataset which was present in the discussions section of the competition.

* Transaction table
“It contains money transfer and also other gifting goods and service, like you booked a ticket for others, etc.”

* TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
“TransactionDT first value is 86400, which corresponds to the number of seconds in a day (60 * 60 * 24 = 86400) so I think the unit is seconds. Using this, we know the data spans 6 months, as the maximum value is 15811131, which would correspond to day 183.”

* TransactionAMT: transaction payment amount in USD
“Some of the transaction amounts have three decimal places to the right of the decimal point. There seems to be a link to three decimal places and a blank addr1 and addr2 field. Is it possible that these are foreign transactions and that, for example, the 75.887 in row 12 is the result of multiplying a foreign currency amount by an exchange rate?”

* ProductCD: product code, the product for each transaction
“Product isn't necessary to be a real 'product' (like one item to be added to the shopping cart). It could be any kind of service.”

* card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.

* addr: address
“both addresses are for purchaser
addr1 as billing region
addr2 as billing country”

* dist: distance
"distances between (not limited) billing address, mailing address, zip code, IP address, phone area, etc.”

* P_ and (R_) emaildomain: purchaser and recipient email domain “ certain transactions don't need recipient, so Remaildomain is null.”

* C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
“Can you please give more examples of counts in the variables C1-15? Would these be like counts of phone numbers, email addresses, names associated with the user? I can't think of 15.
Your guess is good, plus like device, ipaddr, billingaddr, etc. Also these are for both purchaser and recipient, which doubles the number.”

* D1-D15: timedelta, such as days between previous transaction, etc.

* M1-M9: match, such as names on card and address, etc.

* Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.
“For example, how many times the payment card associated with a IP and email or address appeared in 24 hours time range, etc.”
"All Vesta features were derived as numerical. some of them are count of orders within a clustering, a time-period or condition, so the value is finite and has ordering (or ranking). I wouldn't recommend to treat any of them as categorical. If any of them resulted in binary by chance, it maybe worth trying."

* Identity Table
Variables in this table are identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions.
They're collected by Vesta’s fraud protection system and digital security partners.
(The field names are masked and pairwise dictionary will not be provided for privacy protection and contract agreement)

“id01 to id11 are numerical features for identity, which is collected by Vesta and security partners such as device rating, ip_domain rating, proxy rating, etc. Also it recorded behavioral fingerprint like account login times/failed to login times, how long an account stayed on the page, etc. All of these are not able to elaborate due to security partner T&C. I hope you could get basic meaning of these features, and by mentioning them as numerical/categorical, you won't deal with them inappropriately.”

**Labeling logic 
"The logic of our labeling is define reported chargeback on the card as fraud transaction (isFraud=1) and transactions posterior to it with either user account, email address or billing address directly linked to these attributes as fraud too. If none of above is reported and found beyond 120 days, then we define as legit transaction (isFraud=0).
However, in real world fraudulent activity might not be reported, e.g. cardholder was unaware, or forgot to report in time and beyond the claim period, etc. In such cases, supposed fraud might be labeled as legit, but we never could know of them. Thus, we think they're unusual cases and negligible portion."**

**Please note the last paragraph. Most of the participants solved the wrong problem (although they still got good scores !!). Here we do not have to classify whether a transaction is fraudulent or not, because as per the last para all transactions after a fraud transaction for a client is marked fraudulent. So our main motive should be to identify the client for a particular transaction. I know there would still be many questions in your mind, answers of which you would get ahead.**

## <a name="exploration"></a> Exploration

After running some simple exploration scripts, I quickly found out that there are a few aspects of this dataset which makes the competition extremely challenging.

### Class Distribution is Highly Unbalanced

<figure>
  <a href="/images/wavenet.gif"><img style="display: block; margin: auto;" src="/images/wavenet.gif" alt="Wavenet sampling procedure."></a>
  <figcaption>Animation showing sampling from a WaveNet model. The model predicts the distribution of potential signal values for each timestep, given past signal values.</figcaption>
</figure>

### Different distribution of train and test datasets

Here , I would like to mention some important points. Firstly, based on our experiments we found that there were new users in test dataset (i.e. there were users who did not have a single transaction present in test dataset. Secondly, the transaction dates in train and test datasets were disjoint. We had to predict in the future using the past dataset.

## <a name="feature-engineering"></a> Feature Engineering

As alluded to earlier, rather than learning high-level representations of music audio from data, we could also **use existing high-level representations such as MIDI** to construct a hierarchical model. We can use a powerful language model to model music in the symbolic domain, and also construct a conditional WaveNet model that generates audio, given a MIDI representation. Together with my colleagues from the Magenta team at Google AI, [we trained such models](https://magenta.tensorflow.org/maestro-wave2midi2wave) on a new dataset called MAESTRO, which features 172 hours of virtuosic piano performances, captured with fine alignment between note labels and audio waveforms[^maestro]. This dataset is [available to download](https://magenta.tensorflow.org/datasets/maestro) for research purposes.

Compared to hierarchical WaveNets with learnt intermediate representations, this approach yields much better samples in terms of musical structure, but it is limited to instruments and styles of music that MIDI can accurately represent. Manzelli et al. [have demonstrated this approach](http://people.bu.edu/bkulis/projects/music/index.html) for a few instruments other than piano[^manzellithakkar], but the lack of available aligned data could pose a problem.

<figure>
  <img src="/images/wave2midi2wave.png" alt="Wave2Midi2Wave: a transcription model to go from audio to MIDI, a transformer to model MIDI sequences and a WaveNet to synthesise audio given a MIDI sequence.">
  <figcaption>Wave2Midi2Wave: a transcription model to go from audio to MIDI, a transformer to model MIDI sequences and a WaveNet to synthesise audio given a MIDI sequence.</figcaption>
</figure>

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>Wave2Midi2Wave</strong>: <a href="https://openreview.net/forum?id=r1lYRjC9F7">paper</a> - <a href="https://magenta.tensorflow.org/maestro-wave2midi2wave">blog post</a> - <a href="https://storage.googleapis.com/magentadata/papers/maestro/index.html">samples</a> - <a href="https://magenta.tensorflow.org/datasets/maestro">dataset</a><br>
<strong>Manzelli et al. model</strong>: <a href="https://arxiv.org/abs/1806.09905">paper</a> - <a href="http://people.bu.edu/bkulis/projects/music/index.html">samples</a>
</p>

### Sparse transformers

OpenAI introduced the [Sparse Transformer](https://openai.com/blog/sparse-transformer/) model[^sparsetransformer], a large transformer[^transformer] with a **sparse attention mechanism** that scales better to long sequences than traditional attention (which is quadratic in the length of the modelled sequence). They demonstrated impressive results autoregressively modelling language, images, and music audio using this architecture, with sparse attention enabling their model to cope with waveforms of up to 65k timesteps (about 5 seconds at 12 kHz). The sparse attention mechanism seems like a good alternative to the stacked dilated convolutions of WaveNets, provided that an efficient implementation is available.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>Sparse Transformer</strong>: <a href="https://arxiv.org/abs/1904.10509">paper</a> - <a href="https://openai.com/blog/sparse-transformer/">blog post</a> - <a href="https://soundcloud.com/openai_audio/sets/sparse_transformers">samples</a>
</p>

### Universal music translation network

An interesting conditional waveform modelling problem is that of "music translation" or "music style transfer": given a waveform, **render a new waveform where the same music is played by a different instrument**. The Universal Music Translation Network[^umtn] tackles this by training an autoencoder with multiple WaveNet decoders, where the encoded representation is encouraged to be agnostic to the instrument of the input (using an adversarial loss). A separate decoder is trained for each target instrument, so once this representation is extracted from a waveform, it can be synthesised in an instrument of choice. The separation is not perfect, but it works surprisingly well in practice. I think this is a nice example of a model that combines ideas from both likelihood-based models and the adversarial learning paradigm.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>Universal music translation network</strong>: <a href="https://openreview.net/forum?id=HJGkisCcKm">paper</a> - <a href="https://github.com/facebookresearch/music-translation">code</a> - <a href="https://musictranslation.github.io/">samples</a>
</p>

### Dadabots

[Dadabots](http://dadabots.com) are a researcher / artist duo who have trained SampleRNN models on various albums (primarily metal) in order to produce more music in the same vein. These models aren't great at capturing long-range correlations, so it works best for artists whose style is naturally a bit disjointed. Below is a 24 hour livestream they've set up with a model generating infinite technical death metal in the style of 'Relentless Mutation' by Archspire.

<iframe width="560" height="315" src="https://www.youtube.com/embed/MwtVkPKx3RA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## <a name="adversarial-models"></a> Adversarial models of waveforms

Adversarial modelling of audio has only recently started to see some successes, which is why this section is going to be a lot shorter than the previous one on likelihood-based models. The adversarial paradigm has been extremely successful in the image domain, but researchers have had a harder time translating that success to other domains and modalities, compared to likelihood-based models. As a result, published work so far has primarily focused on speech generation and the generation of individual notes or very short clips of music. As a field, we are still very much in the process of figuring out how to make GANs work well for audio at scale.

### WaveGAN

One of the first works to attempt using GANs for modelling raw audio signals is WaveGAN[^wavegan]. They trained a GAN on single-word speech recordings, bird vocalisations, individual drum hits and short excerpts of piano music. They also compared their raw audio-based model with a spectrogram-level model called SpecGAN. Although the fidelity of the [resulting samples](https://chrisdonahue.com/wavegan_examples/) is far from perfect in some cases, this work undoubtedly inspired a lot of researchers to take audio modelling with GANs more seriously.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>WaveGAN</strong>: <a href="https://openreview.net/forum?id=ByMVTsR5KQ">paper</a> - <a href="https://github.com/chrisdonahue/wavegan">code</a> - <a href="https://chrisdonahue.com/wavegan_examples/">samples</a> - <a href="https://chrisdonahue.com/wavegan/">demo</a> - <a href="https://colab.research.google.com/drive/1e9o2NB2GDDjadptGr3rwQwTcw-IrFOnm">colab</a>
</p>

### GANSynth

So far in this blog post, we have focused on generating audio waveforms directly. However, I don't want to omit GANSynth[^gansynth], even though technically speaking it does not operate directly in the waveform domain. This is because the spectral representation it uses is **exactly invertible** -- no other models or phase reconstruction algorithms are used to turn the spectograms it generates into waveforms, which means it shares a lot of the advantages of models that operate directly in the waveform domain.

As <a href="#why-waveforms">discussed before</a>, modelling the phase component of a complex spectrogram is challenging, because the phase of real audio signals can seem essentially random. However, using some of its unique characteristics, we can transform the phase into a quantity that is easier to model and reason about: the *instantaneous frequency*. This is obtained by computing the temporal difference of the *unwrapped* phase between subsequent frames. "Unwrapping" means that we shift the phase component by a multiple of $$2 \pi$$ for each frame as needed to make it monotonic over time, as shown in the diagram below (because phase is an angle, all values modulo $$2 \pi$$ are equivalent).

**The instantaneous frequency captures how much the phase of a signal moves from one spectrogram frame to the next**. For harmonic sounds, this quantity is expected to be constant over time, as the phase rotates at a constant velocity. This makes this representation particularly suitable to model musical sounds, which have a lot of harmonic content (and in fact, it might also make the representation less suitable for modelling more general classes of audio signals, though I don't know if anyone has tried). For harmonic sounds, the instantaneous frequency is almost trivial to predict.

GANSynth is an adversarial model trained to produce the magnitude and instantaneous frequency spectrograms of recordings of individual musical notes. The trained model is also able to generalise to sequences of notes to some degree. [Check out the blog post](https://magenta.tensorflow.org/gansynth) for sound examples and more information.

<figure>
  <img src="/images/gansynth1.png" alt="Waveform with specrogram frame boundaries indicated as dotted lines.">
  <img src="/images/gansynth2.png" alt="From phase to instantaneous frequency.">
  <img src="/images/gansynth3.png" alt="Visualisations of the magnitude, phase, unwrapped phase and instantaneous frequency spectra of a real recording of a note.">
  <figcaption><strong>Top</strong>: waveform with specrogram frame boundaries indicated as dotted lines. <strong>Middle</strong>: from phase to instantaneous frequency. <strong>Bottom</strong>: visualisations of the magnitude, phase, unwrapped phase and instantaneous frequency spectra of a real recording of a note.</figcaption>
</figure>

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>GANSynth</strong>: <a href="https://openreview.net/forum?id=H1xQVn09FX">paper</a> - <a href="http://goo.gl/magenta/gansynth-code">code</a> - <a href="http://goo.gl/magenta/gansynth-examples">samples</a> - <a href="https://magenta.tensorflow.org/gansynth">blog post</a> - <a href="http://goo.gl/magenta/gansynth-demo">colab</a>
</p>

### <a name="melgan-gantts"></a> MelGAN & GAN-TTS

Two recent papers demonstrate excellent results using GANs for text-to-speech: MelGAN[^melgan] and GAN-TTS[^gantts]. The former also includes some music synthesis results, although fidelity is still an issue in that domain. The focus of MelGAN is inversion of magnitude spectrograms (potentially generated by other models), whereas as GAN-TTS is conditioned on the same "linguistic features" as the original WaveNet for TTS.

The architectures of both models share some interesting similarities, which shed light on the right inductive biases for raw waveform discriminators. Both models use **multiple discriminators at different scales**, each of which operates on a **random window** of audio extracted from the full sequence produced by the generator. This is similar to the patch-based discriminators that have occasionally been used in GANs for image generation. This windowing strategy seems to dramatically improve the capability of the generator to **correctly model high frequency content** in the audio signals, which is much more crucial to get right for audio than for images because it more strongly affects perceptual quality. The fact that both models benefited from this particular discriminator design indicates that we may be on the way to figuring out how to best design discriminator architectures for raw audio.

There are also some interesting differences: where GAN-TTS uses a combination of conditional and unconditional discriminators, MelGAN uses only unconditional discriminators and instead encourages the generator output to match the ground truth audio by adding an additional *feature matching* loss: the L1 distance between discriminator feature maps of real and generated audio. Both approaches seem to be effective.

Adversarial waveform synthesis is particularly useful for TTS, because it enables the use of highly parallelisable feed-forward models, which tend to have relatively low capacity requirements because they are trained with a mode-seeking loss. This means the models **can more easily be deployed on low-power hardware while still performing audio synthesis in real-time**, compared to autoregressive or flow-based models.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>MelGAN</strong>: <a href="https://papers.nips.cc/paper/9629-melgan-generative-adversarial-networks-for-conditional-waveform-synthesis">paper</a> - <a href="https://github.com/descriptinc/melgan-neurips">code</a> - <a href="https://melgan-neurips.github.io/">samples</a><br>
<strong>GAN-TTS</strong>: <a href="https://openreview.net/forum?id=r1gfQgSFDr">paper</a> - <a href="https://github.com/mbinkowski/DeepSpeechDistances">code (FDSD)</a> - <a href="https://storage.googleapis.com/deepmind-media/research/abstract.wav">sample</a>
</p>

## <a name="discussion"></a> Discussion

To wrap up this blog post, I want to summarise a few thoughts about the current state of this area of research, and where things could be moving next.

### Why the emphasis on likelihood in music modelling?

Clearly, the dominant paradigm for generative models of music in the waveform domain is likelihood-based. This stands in stark contrast to the image domain, where adversarial approaches greatly outnumber likelihood-based ones. I suspect there are a few reasons for this (let me know if you think of any others):

* Compared to likelihood-based models, it seems like it has been harder to translate the successes of adversarial models in the image domain to other domains, and to the audio domain in particular. I think this is because in a GAN, the discriminator fulfills the role of a **domain-specific loss function**, and important prior knowledge that guides learning is encoded in its architecture. We have known about good architectural priors for images for a long time (stacks of convolutions), as evidenced by work on e.g. style transfer[^styletransfer] and the deep image prior[^deepimageprior]. For other modalities, we don't know as much yet. It seems we are now starting to figure out what kind of architectures work for waveforms (see <a href="#melgan-gantts">MelGAN and GAN-TTS</a>, some relevant work has also been done in the discriminative setting[^randomcnn]).

* **Adversarial losses are mode-seeking**, which makes them more suitable for settings where realism is more important than diversity (for example, because the conditioning signal contains most of the required diversity, as in TTS). In music generation, which is primarily a creative application, **diversity is very important**. Improving diversity of GAN samples is the subject of intense study right now, but I think it could be a while before they catch up with likelihood-based models in this sense.

* The current disparity could also simply be a consequence of the fact that **likelihood-based models got a head start** in waveform modelling, with WaveNet and SampleRNN appearing on the scene in 2016 and WaveGAN in 2018.

Another domain where likelihood-based models dominate is language modelling. I believe the underlying reasons for this might be a bit different though: language is inherently **discrete**, and extending GANs to modelling discrete data at scale is very much a work in progress. This is also more likely to be the reason why likelihood-based models are dominant for symbolic music generation as well: most symbolic representations of music are discrete.

### <a name="alternatives"></a> Alternatives to modelling waveforms directly

Instead of modelling music in the waveform domain, there are many possible alternative approaches. We could model other representations of audio signals, such as spectrograms, as long as we have a way to obtain waveforms from such representations. We have quite a few options for this:

* We could use **invertible spectrograms** (i.e. phase information is not discarded), but in this case modelling the phase poses a considerable challenge. There are ways to make this easier, such as the instantaneous frequency representation used by GANSynth.

* We could also use **magnitude spectrograms** (as is typically done in discriminative models of audio), and then use a **phase reconstruction algorithm** such as the Griffin-Lim algorithm[^griffinlim] to infer a plausible phase component, based only on the generated magnitude. This approach was used for the original Tacotron model for TTS[^tacotron], and for MelNet[^melnet], which models music audio autoregressively in the spectrogram domain.

* Instead of a traditional phase reconstruction algorithm, we could also use a **vocoder** to go from spectrograms to waveforms. A vocoder, in this context, is simply a generative model in the waveform domain, conditioned on spectrograms. Vocoding is a densely conditioned generation task, and many of the models discussed before can and have been used as vocoders (e.g. WaveNet in Tacotron 2[^tacotron2], flow-based models of waveforms, or MelGAN). This approach has some advantages: generated magnitude spectrograms are often imperfect, and vocoder models can learn to account for these imperfections. Vocoders can also work with inherently lossy spectrogram representations such as mel-spectrograms and constant-Q spectrograms[^constantq].
 
* If we are generating audio conditioned on an existing audio signal, we could also simply **reuse the phase** of the input signal, rather than reconstructing or generating it. This is commonly done in source separation, and the approach could also be used for music style transfer.

That said, modelling spectrograms **isn't always easier** than modelling waveforms. Although spectrograms have a much lower temporal resolution, they contain much more information per timestep. In autoregressive models of spectrograms, one would have to condition along both the time and frequency axes to capture all dependencies, which means we end up with roughly as many sequential sampling steps as in the raw waveform case. This is the approach taken by MelNet.

An alternative is to make an **assumption of independence between different frequency bands at each timestep**, given previous timesteps. This enables autoregressive models to produce entire spectrogram frames at a time. This partial independence assumption turns out to be an acceptable compromise in the text-to-speech domain, and is used in Tacotron and Tacotron 2. Vocoder models are particularly useful here as they can attempt to fix the imperfections resulting from this simplification of the model. I'm not sure if anybody has tried, but I would suspect that this independence assumption would cause more problems for music generation.

An interesting new approach combining traditional signal processing ideas with neural networks is [Differentiable Digital Signal Processing (DDSP)](https://magenta.tensorflow.org/ddsp)[^ddsp]. By creating learnable versions of existing DSP components and incorporating them directly into neural networks, these models are endowed with **much stronger inductive biases about sound and music**, and can learn to produce realistic audio with fewer trainable parameters, while also being more interpretable. I suspect that this research direction may gain a lot of traction in the near future, not in the least because the authors [have made their code publicly available](https://github.com/magenta/ddsp), and also because of its modularity and lower computational requirements.

<figure>
  <img src="/images/ddsp.png" alt="Diagram of an example DDSP model. The yellow boxes represent differentiable signal processing components.">
  <figcaption>Diagram of an example DDSP model. The yellow boxes represent differentiable signal processing components. Taken from <a href="https://magenta.tensorflow.org/ddsp">the original blog post</a>.</figcaption>
</figure>

Finally, we could train **symbolic models of music** instead: for many instruments, we already have realistic synthesisers, and we can even train them given enough data (see <a href="#wave2midi2wave">Wave2Midi2Wave</a>). If we are able to craft symbolic representations that capture the aspects of music we care about, then this is an attractive approach as it is much less computationally intensive. Magenta's [Music Transformer](https://magenta.tensorflow.org/music-transformer)[^musictransformer] and OpenAI's [MuseNet](https://openai.com/blog/musenet/) are two models that have recently shown impressive results in this domain, and it is likely that other ideas from the language modelling community could bring further improvements.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>DDSP</strong>: <a href="https://openreview.net/forum?id=B1x1ma4tDr">paper</a> - <a href="https://github.com/magenta/ddsp">code</a> - <a href="https://g.co/magenta/ddsp-examples">samples</a> - <a href="https://magenta.tensorflow.org/ddsp">blog post</a> - <a href="https://g.co/magenta/ddsp-demo">colab</a><br>
<strong>Music Transformer</strong>: <a href="https://openreview.net/forum?id=rJe4ShAcF7">paper</a> - <a href="https://magenta.tensorflow.org/music-transformer">blog post</a><br>
<strong>MuseNet</strong>: <a href="https://openai.com/blog/musenet/">blog post</a>
</p>

### What's next?

Generative models of music in the waveform domain have seen substantial progress over the past few years, but the best results so far are still relatively easy to distinguish from real recordings, even at fairly short time scales. There is still a lot of room for improvement, but I believe a lot of this will be driven by better availability of computational resources, and not necessarily by radical innovation on the modelling front -- we have great tools already, they are simply a bit expensive to use due to **substantial computational requirements**. As time goes on and computers get faster, hopefully this task will garner interest as it becomes accessible to more researchers.

One interesting question is **whether adversarial models are going to catch up** with likelihood-based models in this domain. I think it is quite likely that GANs, having recently made in-roads in the densely conditioned setting, will gradually be made to work for more sparsely conditioned audio generation tasks as well.  Fully unconditional generation with long-term coherence seems very challenging however, and I suspect that the mode-seeking behaviour of the adversarial loss will make this much harder to achieve. A hybrid model, where a GAN captures local signal structure and another model with a different objective function captures high-level structure and long-term correlations, seems like a sensible thing to build.

**Hierarchy** is a very important prior for music (and, come to think of it, for pretty much anything else we like to model), so models that explicitly incorporate this are going to have a leg up on models that don't -- at the cost of some additional complexity. Whether this additional complexity will always be worth it remains to be seen, but at the moment, this definitely seems to be the case.

At any rate, **splitting up the problem into multiple stages** that can be solved separately has been fruitful, and I think it will continue to be. So far, hierarchical models (with learnt or handcrafted intermediate representations) and spectrogram-based models with vocoders have worked well, but perhaps there are other ways to "divide and conquer". A nice example of a different kind of split in the image domain is the one used in Subscale Pixel Networks[^spn], where separate networks model the most and least significant bits of the image data.

## <a name="conclusion"></a> Conclusion

If you made it to the end of this post, congratulations! I hope I've convinced you that music modelling in the waveform domain is an interesting research problem. It is also **very far from a solved problem**, so there are lots of opportunities for interesting new work. I have probably missed a lot of relevant references, especially when it comes to more recent work. If you know about relevant work that isn't discussed here, feel free to share it in the comments! Questions about this blog post and this line of research are very welcome as well.

<!-- TODO: add some bolded parts to highlight them where it makes sense. -->

## <a name="references"></a> References

[^folkrnn]: Sturm, Santos, Ben-Tal and Korshunova, "[Music transcription modelling and composition using deep learning](https://arxiv.org/pdf/1604.08723)", Proc. 1st Conf. Computer Simulation of Musical Creativity, Huddersfield, UK, July 2016. [folkrnn.org](https://folkrnn.org/)

[^pixelrnn]: Van den Oord, Kalchbrenner and Kavukcuoglu, "[Pixel recurrent neural networks](https://arxiv.org/abs/1601.06759)", International Conference on Machine Learning, 2016.

[^pixelcnn]: Van den Oord, Kalchbrenner, Espeholt, Vinyals and Graves, "[Conditional image generation with pixelcnn decoders](http://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders)", Advances in neural information processing systems 29 (NeurIPS), 2016.

[^nice]: Dinh, Krueger and Bengio, "[NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516)", arXiv, 2014.

[^realnvp]: Dinh, Sohl-Dickstein and Bengio, "[Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)", arXiv, 2016.

[^vaekingma]: Kingma and Welling, "[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)", International Conference on Learning Representations, 2014.

[^vaerezende]: Rezende, Mohamed and Wierstra, "[Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082)", International Conference on Machine Learning, 2014.

[^pc]: Bowman, Vilnis, Vinyals, Dai, Jozefowicz and Bengio, "[Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)", 20th SIGNLL Conference on Computational Natural Language Learning, 2016.

[^gans]: Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville and Bengio, "[Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets)", Advances in neural information processing systems 27 (NeurIPS), 2014.

[^aiqn]: Ostrovski, Dabney and Munos, "[Autoregressive Quantile Networks for Generative Modeling](https://arxiv.org/abs/1806.05575)", International Conference on Machine Learning, 2018.

[^scorematching]: Hyvärinen, "[Estimation of Non-Normalized Statistical Models by Score Matching](http://www.jmlr.org/papers/v6/hyvarinen05a.html)", Journal of Machine Learning Research, 2005.

[^energy]: Du and Mordatch, "[https://arxiv.org/abs/1903.08689](https://arxiv.org/abs/1903.08689)", arXiv, 2019.

[^wgan]: Arjovsky, Chintala and Bottou, "[Wasserstein GAN](https://arxiv.org/abs/1701.07875)", arXiv, 2017.

[^swa]: Kolouri, Pope, Martin and Rohde, "[Sliced-Wasserstein Autoencoder: An Embarrassingly Simple Generative Model](https://arxiv.org/abs/1804.01947)", arXiv, 2018.

[^ssm]: Song, Garg, Shi and Ermon, "[Sliced Score Matching: A Scalable Approach to Density and Score Estimation](https://arxiv.org/abs/1905.07088)", UAI, 2019.

[^scorebased]: Song and Ermon, "[Generative Modeling by Estimating Gradients of the Data Distribution](http://papers.nips.cc/paper/9361-generative-modeling-by-estimating-gradients-of-the-data-distribution)", Advances in neural information processing systems 32 (NeurIPS), 2019.

[^wavenet]: Van den Oord, Dieleman, Zen, Simonyan, Vinyals, Graves, Kalchbrenner, Senior and Kavukcuoglu, "[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)", arXiv, 2016.

[^samplernn]: Mehri, Kumar, Gulrajani, Kumar, Jain, Sotelo, Courville and Bengio, "[SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837)", International Conference on Learning Representations, 2017.

[^parallelwavenet]: Van den Oord, Li, Babuschkin, Simonyan, Vinyals, Kavukcuoglu, van den Driessche, Lockhart, Cobo, Stimberg, Casagrande, Grewe, Noury, Dieleman, Elsen, Kalchbrenner, Zen, Graves, King, Walters, Belov and Hassabis, "[Parallel WaveNet: Fast High-Fidelity Speech Synthesis](https://arxiv.org/abs/1711.10433)", International Conference on Machine Learning, 2018.

[^clarinet]: Ping, Peng and Chen, "[ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech](https://arxiv.org/abs/1807.07281)", International Conference on Learning Representations, 2019.

[^waveglow]: Prenger, Valle and Catanzaro, "[WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)", International Conference on Acoustics, Speech, and Signal Procesing, 2019

[^flowavenet]: Kim, Lee, Song, Kim and Yoon, "[FloWaveNet : A Generative Flow for Raw Audio](https://arxiv.org/abs/1811.02155)", International Conference on Machine Learning, 2019.

[^waveflow]: Ping, Peng, Zhao and Song, "[WaveFlow: A Compact Flow-based Model for Raw Audio](https://arxiv.org/abs/1912.01219)", ArXiv, 2019.

[^blow]: Serrà, Pascual and Segura, "[Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion](https://papers.nips.cc/paper/8904-blow-a-single-scale-hyperconditioned-flow-for-non-parallel-raw-audio-voice-conversion)", Advances in neural information processing systems 32 (NeurIPS), 2019.

[^vqvae]: Van den Oord, Vinyals and Kavukcuoglu, "[Neural Discrete Representation Learning](http://papers.nips.cc/paper/7210-neural-discrete-representation-learning)", Advances in neural information processing systems 30 (NeurIPS), 2017.

[^challenge]: Dieleman, Van den Oord and Simonyan, "[The challenge of realistic music generation: modelling raw audio at scale](https://papers.nips.cc/paper/8023-the-challenge-of-realistic-music-generation-modelling-raw-audio-at-scale)", Advances in neural information processing systems 31 (NeurIPS), 2018.

[^maestro]: Hawthorne, Stasyuk, Roberts, Simon, Huang, Dieleman, Elsen, Engel and Eck, "[Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset](https://openreview.net/forum?id=r1lYRjC9F7)", International Conference on Learning Representations, 2019.

[^manzellithakkar]: Manzelli, Thakkar, Siahkamari and Kulis, "[Conditioning Deep Generative Raw Audio Models for Structured Automatic Music](https://arxiv.org/abs/1806.09905)", International Society for Music Information Retrieval Conference, 2018.

[^sparsetransformer]: Child, Gray, Radford and Sutskever, "[Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)", Arxiv, 2019.

[^transformer]: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin, "[Attention is All you Need](http://papers.nips.cc/paper/7181-attention-is-all-you-need)", Advances in neural information processing systems 30 (NeurIPS), 2017.

[^umtn]: Mor, Wolf, Polyak and Taigman, "[A Universal Music Translation Network](https://openreview.net/forum?id=HJGkisCcKm)", International Conference on Learning Representations, 2019.

[^wavegan]: Donahue, McAuley and Puckette, "[Adversarial Audio Synthesis](https://openreview.net/forum?id=ByMVTsR5KQ)", International Conference on Learning Representations, 2019.

[^gansynth]: Engel, Agrawal, Chen, Gulrajani, Donahue and Roberts, "[GANSynth: Adversarial Neural Audio Synthesis](https://openreview.net/forum?id=H1xQVn09FX)", International Conference on Learning Representations, 2019.

[^melgan]: Kumar, Kumar, de Boissiere, Gestin, Teoh, Sotelo, de Brébisson, Bengio and Courville, "[MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://papers.nips.cc/paper/9629-melgan-generative-adversarial-networks-for-conditional-waveform-synthesis)", Advances in neural information processing systems 32 (NeurIPS), 2019.

[^gantts]: Bińkowski, Donahue, Dieleman, Clark, Elsen, Casagrande, Cobo and Simonyan, "[High Fidelity Speech Synthesis with Adversarial Networks](https://openreview.net/forum?id=r1gfQgSFDr)", International Conference on Learning Representations, 2020.

[^styletransfer]: Gatys, Ecker and Bethge, "[Image Style Transfer Using Convolutional Neural Networks](http://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)", IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[^deepimageprior]: Ulyanov, Vedaldi and Lempitsky, "[Deep Image Prior](http://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html)", IEEE Conference on Computer Vision and Pattern Recognition, 2018.

[^randomcnn]: Pons and Serra, "[Randomly weighted CNNs for (music) audio classification](https://arxiv.org/abs/1805.00237)", IEEE International Conference on Acoustics, Speech and Signal Processing, 2019.

[^griffinlim]: Griffin and Lim, "[Signal estimation from modified short-time Fourier transform](https://ieeexplore.ieee.org/abstract/document/1164317/)", IEEE Transactions on Acoustics, Speech and Signal Processing, 1984.

[^tacotron]: Wang, Skerry-Ryan, Stanton, Wu, Weiss, Jaitly, Yang, Xiao, Chen, Bengio, Le, Agiomyrgiannakis, Clark and Saurous, "[Tacotron: Towards end-to-end speech synthesis](https://arxiv.org/abs/1703.10135)", Interspeech, 2017.

[^melnet]: Vasquez and Lewis, "[Melnet: A generative model for audio in the frequency domain](https://arxiv.org/abs/1906.01083)", ArXiv, 2019.

[^tacotron2]: Shen, Pang, Weiss, Schuster, Jaitly, Yang, Chen, Zhang, Wang, Skerry-Ryan, Saurous, Agiomyrgiannakis, Wu, "[Natural TTS synthesis by conditioning wavenet on mel spectrogram predictions](https://arxiv.org/abs/1712.05884)", IEEE International Conference on Acoustics, Speech and Signal Processing, 2018.

[^constantq]: Schörkhuber and Klapuri, "[Constant-Q transform toolbox for music processing](https://iem.kug.ac.at/fileadmin/media/iem/projects/2010/smc10_schoerkhuber.pdf)", Sound and Music Computing Conference, 2010.

[^ddsp]: Engel, Hantrakul, Gu and Roberts, "[DDSP: Differentiable Digital Signal Processing](https://openreview.net/forum?id=B1x1ma4tDr)", International Conference on Learning Representations, 2020.

[^musictransformer]: Huang, Vaswani, Uszkoreit, Simon, Hawthorne, Shazeer, Dai, Hoffman, Dinculescu and Eck, "[Music Transformer: Generating Music with Long-Term Structure ](https://openreview.net/forum?id=rJe4ShAcF7)", International Conference on Learning Representations, 2019.

[^spn]: Menick and Kalchbrenner, "[Generating High Fidelity Images with Subscale Pixel Networks and Multidimensional Upscaling](https://openreview.net/forum?id=HylzTiC5Km)", International Conference on Learning Representations, 2019.
