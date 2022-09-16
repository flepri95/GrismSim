import numpy as np
from scipy import optimize, sparse, linalg
from tqdm.auto import tqdm

import utils
import galaxy
import consts

def make_gaussian_kernel(shape, nlines, n=3, sigma=1):
    offsets = np.arange(-n, n+1)
    values = np.exp(-0.5*offsets**2/sigma**2)
    values /= np.sum(values)
    #print("v", np.sum(values))
    d = []
    for i in range(len(offsets)):
        d.append(np.ones(shape[0]) * values[i])
    m = sparse.dia_array((d, offsets), shape)
    m = m.tolil()
    '''m[shape[0]-1,:] = 0
    m[:, shape[1]-1] = 0
    m[-1,-1] = 1'''
    for i in range(nlines):
        m[-(i+1),:] = 0
        m[:, -(i+1)] = 0

    for i in range(nlines):
        m[-(i+1),-(i+1)] = 1

    m= m.tocsr()
    return m


def pseudoinv(Q, r=1e-10, k=None):
    """ """
    if k is None:
        k = Q.shape[0]-1

    if k >= Q.shape[0]:
        k = Q.shape[0]-1

    u, s, v = sparse.linalg.svds(Q, k=k)

    sinv = np.zeros(len(s))

    threshold = s.max()*r

    nonzero = s > threshold
    #print(f"number of singular values {np.sum(nonzero)} of {Q.shape[0]}")

    sinv[nonzero] = 1./s[nonzero]

    Qinv = np.dot(v.T*sinv, u.T)

    return Qinv


def Solve(A, b, b_var, nlines, k=None, return_cov=False):
    """ """
    inv_var = 1./b_var

    #inv_var=np.ones_like(b_var)

    Q = (A.T * inv_var) @ A

    # Qinv = pseudoinv(Q, k=k)
    try:
        Qinv = linalg.pinv(Q.toarray())
    except:
        return np.zeros(A.shape[1])

    kernel = make_gaussian_kernel(Q.shape, nlines, 3)

    QinvA = kernel @ Qinv @ (A.T * inv_var)

    #QinvA = Qinv @ (A.T * inv_var)

    x = QinvA @ b
    if return_cov==False:
        return x
    else:
        'Computing Cov'
        cov = (QinvA*b_var) @ QinvA.T
        return x, cov

    return x


def EvaluateModel(sed_params, galaxy, frame_list, mask_list):
    """ """
    galaxy.sed = sed_params*1e-18
    images = []
    for i, frame in enumerate(frame_list):
        images.append(frame.make_image([galaxy], mask=mask_list[i], noise=False, return_var=False))
    return images


def EvaluateEmLineModel(sed_params, fluxes, galaxy, frame_list, mask_list):
    """ """
    galaxy.sed = sed_params*1e-18
    galaxy.params['fluxes_emlines']=fluxes*1e-16
    images = []
    for i, frame in enumerate(frame_list):
        images.append(frame.make_image([galaxy], mask=mask_list[i], noise=False, return_var=False))
    return images


def get_data_vec(data, mask_list):
    """ """
    data_vec = []
    var_vec = []
    for i in range(len(data)):

        sel = mask_list[i].flatten() > -1

        d, v = data[i]
        d = d.flatten()[sel]
        v = v.flatten()[sel]

        data_vec = np.concatenate([data_vec, d])
        var_vec = np.concatenate([var_vec, v])
    return data_vec, var_vec

def ComputeJacobian(galaxy_list, frame_list, mask_list, dx=100):
    """ """
    galaxy_list = utils.ensurelist(galaxy_list)

    npix = 0
    for mask in mask_list:
        npix += np.sum(mask > -1)

    jac_list = []

    for galaxy in galaxy_list:

        jac = sparse.lil_array(
            (npix, len(galaxy.wavelength)),
            dtype='d'
        )

        galaxy.params['fluxes_emlines'] = np.zeros(galaxy.params['nlines'])
        test_sed = np.zeros(len(galaxy.wavelength), dtype='d')

        for i in range(len(galaxy.wavelength)):
            test_sed *= 0
            test_sed[i] = dx

            model = EvaluateModel(test_sed, galaxy, frame_list, mask_list)
            concat_model = np.concatenate(model)

            jac[:, i] = concat_model / dx

        jac_list.append(jac)

    jac = sparse.hstack(jac_list)

    jac = sparse.csr_array(jac)

    print(f"Computing Jacobian done {jac.shape}")

    return jac


def ComputeJacobianEmLine(galaxy, frame_list, mask_list, dx=1000):
    """ """

    test_sed = np.zeros(len(galaxy.wavelength), dtype='d')

    # perturb the emission line by dx
    test_fluxes = np.zeros(galaxy.params['nlines'])

    jac_list = []

    for i in range(galaxy.params['nlines']):

        test_fluxes *= 0
        test_fluxes[i] = dx

        model = EvaluateEmLineModel(test_sed, test_fluxes, galaxy, frame_list, mask_list)
        concat_model = np.concatenate(model)

        jac = concat_model / dx
        jac_list.append(jac)

    jac = np.array(jac_list).T
    jac = sparse.csr_array(jac)

    return jac


def ComputeJacobianEmLine2(galaxy, frame_list, mask_list, coeff, dx=50):
    """ """
    test_sed = np.zeros(len(galaxy.wavelength), dtype='d')

    # perturb the emission line by dx
    test_fluxes = np.zeros(galaxy.params['nlines'])

    for i in range(len(test_fluxes)):
        wavelength_obs = (1+galaxy.params['redshift'])*consts.lines[i]
        if wavelength_obs < galaxy.params['obs_wavelength_range'][0] or wavelength_obs > galaxy.params['obs_wavelength_range'][1]:
            test_fluxes[i] = 0
        else:
            test_fluxes[i] = dx*coeff[i]

    model = EvaluateEmLineModel(test_sed, test_fluxes, galaxy, frame_list, mask_list)

    concat_model = np.concatenate(model)

    jac = concat_model / dx

    jac = np.array(jac).T
    jac = jac.reshape(len(jac), 1)
    jac = sparse.csr_array(jac)

    return jac

def MeasureSource(images, frame_list, mask_list, test_galaxy, redshift_grid, flux_est=False, return_cov=False, k=None, dx=100):
    """Measure a source"""

    test_galaxy = test_galaxy
    extractions = []

    #mask = frame.make_mask(test_galaxy)

    masks_list = []
    for frame_i in range(len(frame_list)):
        mask = frame_list[frame_i].make_mask(test_galaxy, input_mask=mask_list[0])
        masks_list.append(mask)

    jac_con = ComputeJacobian(test_galaxy, frame_list, masks_list)

    # compute jacobian of the continuum model
    # d image/d continuum

    # cutting out the spectrum from the image
    data, var = get_data_vec(images, masks_list)

    #data -= np.median(data)

    pz = np.zeros(len(redshift_grid))
    px = []
    fluxes=[]

    coeff = np.loadtxt('data/spectemplates.txt', skiprows=1, delimiter=',', dtype='float')
    coeff = coeff[1:]

    for i in tqdm(range(len(redshift_grid))):

        # compute jacobian of the emission line flux at redshift z
        test_galaxy.params['redshift'] = redshift_grid[i]
        jac_em = ComputeJacobianEmLine2(test_galaxy, frame_list, masks_list, coeff, dx=dx)

        # concatenate jacobian matrices
        jac = sparse.hstack((jac_con, jac_em))
        jac = sparse.csr_array(jac)

        #x, cov = solve(jac, data, var, k=k)
        x = Solve(jac, data, var, 1, k=k)

        # compute likelihood of best fit
        loglike = -0.5 * (jac @ x - data)**2 / var

        x_c = np.zeros_like(x)
        x_c[:len(test_galaxy.wavelength)] = x[:len(test_galaxy.wavelength)]

        norm = np.sum((jac@x_c - data)**2/var)
        #norm = np.sum(data**2/var)
        #print(loglike.shape)
        logsum = 2*np.sum(loglike) + norm

        pz[i]=logsum
        px.append(x)

    # return best solution that maximizes the loglike

    hyp_z_1 = []
    max_like = np.max(pz)
    min_like = np.min(pz)
    avg_like = np.average(pz)
    p_index=[]

    for i in range(len(redshift_grid)):
        if(pz[i]-avg_like>2/3*(max_like-avg_like)):
            hyp_z_1.append(redshift_grid[i])
            p_index.append(pz[i])
        else:
            continue
    p_index=np.array(p_index)
    hyp_z_1=np.array(hyp_z_1)
    if len(hyp_z_1)>5:
        ind=np.argpartition(p_index, -5)[-5:]
        hyp_z = hyp_z_1[ind]
        print(f'Reduced number of redshifts with refined grid from {len(hyp_z_1)} to 5')
    else:
        hyp_z=hyp_z_1
    print(len(hyp_z))

    new_px=[]
    new_pz=[]
    new_redshifts=np.linspace(min(redshift_grid), max(redshift_grid), len(redshift_grid)*5)
    index=[]

    for z in hyp_z:
        for zi in range(len(new_redshifts)):
            if abs(z-new_redshifts[zi])<=0.015:
                index.append(zi)

    new_redshifts = new_redshifts[index]
    new_redshifts = np.unique(new_redshifts)

    print('Refining solution...')

    for i in tqdm(range(len(new_redshifts))):

        # compute jacobian of the emission line flux at redshift z
        test_galaxy.params['redshift'] = new_redshifts[i]
        jac_em = ComputeJacobianEmLine2(test_galaxy, frame_list, masks_list, coeff)

        # concatenate jacobian matrices
        jac = sparse.hstack((jac_con, jac_em))
        jac = sparse.csr_array(jac)

        #x, cov = solve(jac, data, var, k=k)
        x = Solve(jac, data, var, 1, k=k)

        # compute likelihood of best fit
        loglike = -0.5 * (jac @ x - data)**2 / var
        x_c = np.zeros_like(x)
        x_c[:len(test_galaxy.wavelength)] = x[:len(test_galaxy.wavelength)]

        norm = np.sum((jac@x_c - data)**2/var)
        #norm = np.sum(data**2/var)
        logsum = 2*np.sum(loglike) + norm

        new_pz.append(logsum)
        new_px.append(x)

    new_pz = np.array(new_pz)
    best = np.argmax(new_pz)

    if flux_est == False:

        extr = new_px[best]
        flux_line = extr[-1:] #extr[-test_galaxy.params['nlines']:]
        extr = extr[0:len(test_galaxy.wavelength)]

        return test_galaxy.wavelength, extr, flux_line, pz, new_pz, new_redshifts, new_redshifts[best]

    else:

        ref_redshift = new_redshifts[best]#np.linspace(new_redshifts[best]-0.003, new_redshifts[best]+0.003, 20)
        #ref_pz = []
        #ref_px = []

        #for i in tqdm(range(len(ref_redshifts))):

        # compute jacobian of the emission line flux at redshift z
        test_galaxy.params['redshift'] = ref_redshift#s[i]
        jac_em = ComputeJacobianEmLine(test_galaxy, frame_list, masks_list)

        # concatenate jacobian matrices
        jac = sparse.hstack((jac_con, jac_em))
        jac = sparse.csr_array(jac)

        #x, cov = solve(jac, data, var, k=k)
        if return_cov==False:
            x = Solve(jac, data, var, test_galaxy.params['nlines'], return_cov=return_cov, k=k)
        else:
            x, cov = Solve(jac, data, var, test_galaxy.params['nlines'], return_cov=return_cov, k=k)

        loglike = -0.5 * (jac @ x - data)**2 / var
        x_c = np.zeros_like(x)
        x_c[:len(test_galaxy.wavelength)] = x[:len(test_galaxy.wavelength)]

        norm = np.sum((jac@x_c - data)**2/var)
        #norm = np.sum(data**2/var)
        logsum = 2*np.sum(loglike) + norm

            #ref_pz.append(logsum)
            #ref_px.append(x)

        #ref_pz = np.array(ref_pz)
        #best = np.argmax(ref_pz)

        extr = x#ref_px[best]
        flux_line = extr[-test_galaxy.params['nlines']:]
        extr = extr[0:len(test_galaxy.wavelength)]
        if return_cov==False:
            return test_galaxy.wavelength, extr, flux_line, pz, new_pz, new_redshifts, new_redshifts[best], logsum
        else:
            return test_galaxy.wavelength, extr, flux_line, pz, new_pz, new_redshifts, new_redshifts[best], logsum, cov
        #return test_galaxy.wavelength, extr, flux_line, pz, new_pz, ref_pz, new_redshifts, ref_redshifts, ref_redshifts[best]

def MeasureFlux(images, frame_list, mask_list, test_galaxy, k=None, dx=1000):


    wave = test_galaxy.wavelength
    extractions = []

    #mask = frame.make_mask(test_galaxy)

    masks_list = []
    for frame_i in range(len(frame_list)):
        mask = frame_list[frame_i].make_mask(test_galaxy, input_mask=mask_list[0])
        masks_list.append(mask)

    # compute jacobian of the continuum model
    # d image/d continuum

    jac_con = ComputeJacobian(test_galaxy, frame_list, masks_list)

    # cutting out the spectrum from the image
    data, var = get_data_vec(images, masks_list)

    jac_em = ComputeJacobianEmLine(test_galaxy, frame_list, masks_list, dx=dx)

    # concatenate jacobian matrices
    jac = sparse.hstack((jac_con, jac_em))
    jac = sparse.csr_array(jac)

    #x, cov = solve(jac, data, var, k=k)
    x = Solve(jac, data, var, test_galaxy.params['nlines'], k=k)

    # compute likelihood of best fit
    loglike = -0.5 * (jac @ x - data)**2 / var

    x_c = np.zeros_like(x)
    x_c[:len(test_galaxy.wavelength)] = x[:len(test_galaxy.wavelength)]

    norm = np.sum((jac@x_c - data)**2/var)
    logsum = 2*np.sum(loglike) + norm

    # return best solution that maximizes the loglike

    extr = x#ref_px[best]
    flux_line = extr[-test_galaxy.params['nlines']:]
    extr = extr[0:len(test_galaxy.wavelength)]

    return test_galaxy.wavelength, extr, flux_line, logsum

def MeasureEverything(images, frame_list, mask_list, galaxy_list, redshift_grid, flux_est=False, k=None):
    """Measure all sources"""
    
    for gal_i in galaxy_list:
        gal_i.params['fluxes_emlines']=np.zeros(gal_i.params['nlines'])
        
    wave = galaxy_list[0].wavelength

    masks_list = []
    for frame_i in range(len(frame_list)):
        mask = frame_list[frame_i].make_mask(galaxy_list, input_mask=mask_list[0])
        masks_list.append(mask)

    jac_con = ComputeJacobian(galaxy_list, frame_list, masks_list)

    coeff = np.loadtxt('data/spectemplates.txt', skiprows=1, delimiter=',', dtype='float')
    coeff = coeff[1:]

    data, var = get_data_vec(images, masks_list)

    z_evals=[]
    loglike_all=[]
    loglike_ref=[]
    loglike_flux=[]
    extractions=[]
    fluxes=[]
    ref_redshifts=[]
    ref_flux_redshifts=[]

    for gal_i in tqdm(range(len(galaxy_list))):

        pz = np.zeros(len(redshift_grid))
        px = []

        for i in tqdm(range(len(redshift_grid))):

            # compute jacobian of the emission line flux at redshift z
            galaxy_list[gal_i].params['redshift'] = redshift_grid[i]
            jac_em = ComputeJacobianEmLine2(galaxy_list[gal_i], frame_list, masks_list, coeff)

            # concatenate jacobian matrices

            jac = sparse.hstack((jac_con, jac_em))
            jac = sparse.csr_array(jac)
            
            #x, cov = solve(jac, data, var, k=k)
            x = Solve(jac, data, var, galaxy_list[gal_i].params['nlines'], k=k)
            
            loglike = -0.5 * (jac @ x - data)**2 / var
            x_c = np.zeros_like(x)
            x_c[:len(galaxy_list[gal_i].wavelength)*len(galaxy_list)] = x[:len(galaxy_list[gal_i].wavelength)*len(galaxy_list)]

            norm = np.sum((jac@x_c - data)**2/var)
            #norm = np.sum(data**2/var)
            logsum = 2*np.sum(loglike) + norm

            pz[i]=logsum
            px.append(x)

        hyp_z_1 = []
        max_like = np.max(pz)
        min_like = np.min(pz)
        avg_like = np.average(pz)
        p_index=[]

        for i in range(len(redshift_grid)):
            if(pz[i]-avg_like>2/3*(max_like-avg_like)):
                hyp_z_1.append(redshift_grid[i])
                p_index.append(pz[i])
            else:
                continue
                
        p_index=np.array(p_index)
        hyp_z_1=np.array(hyp_z_1)
        if len(hyp_z_1)>5:
            ind=np.argpartition(p_index, -5)[-5:]
            hyp_z = hyp_z_1[ind]
            print(f'Reduced number of redshifts with refined grid from {len(hyp_z_1)} to 5')
        else:
            hyp_z=hyp_z_1

        print(len(hyp_z))

        new_px=[]
        new_pz=[]
        new_redshifts=np.linspace(min(redshift_grid), max(redshift_grid), len(redshift_grid)*5)
        index=[]

        for z in hyp_z:
            for zi in range(len(new_redshifts)):
                if abs(z-new_redshifts[zi])<=0.015:
                    index.append(zi)
        new_redshifts = new_redshifts[index]
        new_redshifts = np.unique(new_redshifts)

        print('Refining solution...')

        for i in tqdm(range(len(new_redshifts))):

            # compute jacobian of the emission line flux at redshift z
            galaxy_list[gal_i].params['redshift'] = new_redshifts[i]
            jac_em = ComputeJacobianEmLine2(galaxy_list[gal_i], frame_list, masks_list, coeff)

            # concatenate jacobian matrices
            jac = sparse.hstack((jac_con, jac_em))
            jac = sparse.csr_array(jac)

            #x, cov = solve(jac, data, var, k=k)
            x = Solve(jac, data, var, 1, k=k)

            loglike = -0.5 * (jac @ x - data)**2 / var
            x_c = np.zeros_like(x)
            x_c[:len(galaxy_list[gal_i].wavelength)*len(galaxy_list)] = x[:len(galaxy_list[gal_i].wavelength)*len(galaxy_list)]

            norm = np.sum((jac@x_c - data)**2/var)
            #norm = np.sum(data**2/var)
            logsum = 2*np.sum(loglike) + norm

            new_pz.append(logsum)
            new_px.append(x)

        new_pz = np.array(new_pz)

    # return best solution that maximizes the loglike

        best = np.argmax(new_pz)

        if flux_est == False:

            z_evals.append(new_redshifts[best])
            extr = new_px[best]
            flux_line = extr[-1:]#extr[-galaxies[gal_i].params['nlines']:]
            extr = extr[gal_i*len(galaxy_list[gal_i].wavelength):(gal_i+1)*len(galaxy_list[gal_i].wavelength)]
            extractions.append(extr)
            loglike_all.append(pz)
            loglike_ref.append(new_pz)
            fluxes.append(flux_line)
            ref_redshifts.append(new_redshifts)

        else:

            ref_redshift = new_redshifts[best]

            print('Estimating line fluxes...')

            # compute jacobian of the emission line flux at redshift z
            galaxy_list[gal_i].params['redshift'] = ref_redshift
            jac_em = ComputeJacobianEmLine(galaxy_list[gal_i], frame_list, masks_list)

            # concatenate jacobian matrices
            jac = sparse.hstack((jac_con, jac_em))
            jac = sparse.csr_array(jac)

            #x, cov = solve(jac, data, var, k=k)
            x = Solve(jac, data, var, galaxy_list[gal_i].params['nlines'], k=k)

            loglike = -0.5 * (jac @ x - data)**2 / var
            x_c = np.zeros_like(x)
            x_c[:len(galaxy_list[gal_i].wavelength)*len(galaxy_list)] = x[:len(galaxy_list[gal_i].wavelength)*len(galaxy_list)]

            norm = np.sum((jac@x_c - data)**2/var)
            #norm = np.sum(data**2/var)
            logsum = 2*np.sum(loglike) + norm

            z_evals.append(new_redshifts[best])
            extr = x
            flux_line = extr[-galaxy_list[gal_i].params['nlines']:]
            extr = extr[gal_i*len(galaxy_list[gal_i].wavelength):(gal_i+1)*len(galaxy_list[gal_i].wavelength)]
            extractions.append(extr)
            loglike_all.append(pz)
            loglike_ref.append(new_pz)
            loglike_flux.append(logsum)
            fluxes.append(flux_line)
            ref_redshifts.append(new_redshifts)

    if flux_est == False:

        return galaxy_list[0].wavelength, extractions, fluxes, loglike_all, loglike_ref, ref_redshifts, z_evals

    else:

        return galaxy_list[0].wavelength, extractions, fluxes,loglike_all, loglike_ref, loglike_flux, ref_redshifts, z_evals




